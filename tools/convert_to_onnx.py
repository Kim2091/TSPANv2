import torch
import torch.onnx
import argparse
import os
import sys
import numpy as np
import onnx
import logging

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Import the correct model architecture
from models.tspanv2 import TSPANv2

class TemporalSPANExportWrapper(torch.nn.Module):
    """
    A simple wrapper for the TSPANv2 model to handle
    the input shape transformation required for ONNX export. The rest of the
    script expects a 4D tensor, while the model's forward pass expects a 5D tensor.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Get clip_size from the model instance for the reshape operation
        self.num_frames = model.clip_size
        # The number of channels per frame is fixed (e.g., RGB)
        self.channels_per_frame = 3  # RGB channels

    def forward(self, x):
        """
        Takes a 4D tensor and reshapes it for the model.
        Args:
            x: Input tensor with shape (batch, num_frames * channels, height, width)
        """
        # Get dynamic shape info from the input tensor
        b, _, h, w = x.shape
        
        # Reshape to the 5D format the model's forward() method expects
        # (batch, num_frames, channels, height, width)
        reshaped_x = x.view(b, self.num_frames, self.channels_per_frame, h, w)
        
        # Call the original model's forward pass with the correctly shaped tensor
        return self.model(reshaped_x)

def verify_onnx_output(model, onnx_path, test_input, rtol=1e-3, atol=1e-4):
    """
    Verify ONNX model output against PyTorch model output.
    This function remains largely the same but is now called with the wrapper.
    """
    try:
        import onnxruntime as ort
        
        logger.info("\nVerifying ONNX model...")

        # Get PyTorch output using the wrapper
        model.eval()
        with torch.inference_mode():
            torch_output = model(test_input).cpu().numpy()

        # Load and check the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
        # Prepare input for ONNX Runtime
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
        
        # Run ONNX model
        onnx_output = ort_session.run(None, ort_inputs)[0]

        # Compare outputs
        logger.info(f"PyTorch output shape: {torch_output.shape}")
        logger.info(f"ONNX output shape:    {onnx_output.shape}")
        
        # Calculate detailed difference metrics
        abs_diff = np.abs(torch_output - onnx_output)
        rel_diff = abs_diff / (np.abs(torch_output) + 1e-8)
        
        logger.info("\n=== Difference Metrics ===")
        logger.info(f"Absolute difference:")
        logger.info(f"  Mean: {abs_diff.mean():.6e}")
        logger.info(f"  Max:  {abs_diff.max():.6e}")
        logger.info(f"  Min:  {abs_diff.min():.6e}")
        
        logger.info(f"\nRelative (percentage) difference:")
        logger.info(f"  Mean: {rel_diff.mean() * 100:.4f}%")
        logger.info(f"  Max:  {rel_diff.max() * 100:.4f}%")
        logger.info(f"  Min:  {rel_diff.min() * 100:.4f}%")
        
        logger.info(f"\nOutput value ranges:")
        logger.info(f"  PyTorch - Min: {torch_output.min():.6f}, Max: {torch_output.max():.6f}")
        logger.info(f"  ONNX    - Min: {onnx_output.min():.6f}, Max: {onnx_output.max():.6f}")
        
        # Perform assertion
        np.testing.assert_allclose(torch_output, onnx_output, rtol=rtol, atol=atol)
        logger.info("\n✓ ONNX output verified successfully against PyTorch output.")
        logger.info(f"  (within rtol={rtol}, atol={atol})")
        return True
            
    except ImportError:
        logger.warning("⚠ ONNX Runtime not installed. Skipping verification.")
        return False
    except Exception as e:
        logger.error(f"❌ Error during ONNX verification: {str(e)}")
        return False

def export_model_fp16(export_model, dummy_input, output_path, dynamic_axes, device):
    """
    Export model directly to FP16 ONNX by converting PyTorch model to half precision first.
    This is more reliable than post-converting the ONNX graph.
    """
    logger.info(f"\nExporting model to ONNX (FP16): {output_path}")
    
    try:
        # Convert model to FP16
        export_model_fp16 = export_model.half()
        export_model_fp16.eval()
        
        # Create FP16 dummy input
        dummy_input_fp16 = dummy_input.half()
        
        # Export directly to ONNX with FP16 weights
        torch.onnx.export(
            export_model_fp16,
            dummy_input_fp16,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
        )
        
        logger.info(f"✓ Successfully saved FP16 model to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error during FP16 export: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_model_from_state(state_dict, in_nc=3, out_nc=3, clip_size=5, 
                          dim=48, upscale=4, bias=False, num_blocks=6,
                          residual=True, img_range=255., use_checkpoint=False):
    """
    Load a TSPANv2 model from a state dict.
    
    Args:
        state_dict: The state dictionary to load
        in_nc: Number of input channels (default: 3 for RGB)
        out_nc: Number of output channels (default: 3 for RGB)
        clip_size: Number of frames in input clip (must be odd, default: 5)
        dim: Feature dimension (default: 48)
        upscale: Upscaling factor (default: 4)
        bias: Use bias in SPAB blocks (default: False)
        num_blocks: Number of SPAB blocks per temporal layer (default: 6)
        residual: Add residual connection from center frame (default: True)
        img_range: Image range for normalization (default: 255.0)
        use_checkpoint: Use gradient checkpointing (default: False)
    
    Returns:
        Loaded TSPANv2 model
    """
    # Try to infer parameters from state dict if they exist
    if 'params' in state_dict:
        params = state_dict['params']
        in_nc = params.get('in_nc', in_nc)
        out_nc = params.get('out_nc', out_nc)
        clip_size = params.get('clip_size', clip_size)
        dim = params.get('dim', dim)
        upscale = params.get('upscale', upscale)
        bias = params.get('bias', bias)
        num_blocks = params.get('num_blocks', num_blocks)
        residual = params.get('residual', residual)
        img_range = params.get('img_range', img_range)
        state_dict = state_dict['params_ema'] if 'params_ema' in state_dict else state_dict['params']
    elif 'params_ema' in state_dict:
        # Use params_ema if available
        state_dict = state_dict['params_ema']
    
    # Infer scale from upsampler weight shape if available
    if 'm_upsample.0.weight' in state_dict:
        upsampler_weight = state_dict['m_upsample.0.weight']
        # upsampler output channels = dim * (scale^2)
        detected_scale = int((upsampler_weight.shape[0] / dim) ** 0.5)
        logger.info(f"Detected upscale factor from model weights: {detected_scale}x")
        upscale = detected_scale
    
    # Create the model with the correct parameters
    model = TSPANv2(
        in_nc=in_nc,
        out_nc=out_nc,
        clip_size=clip_size,
        dim=dim,
        num_blocks=num_blocks,
        upscale=upscale,
        bias=bias,
        residual=residual,
        img_range=img_range,
        use_checkpoint=use_checkpoint
    )
    
    # Load the state dict
    model.load_state_dict(state_dict, strict=False)
    
    return model

def convert_model_to_onnx(model_path, onnx_path, input_shape, dynamic=False, verify=True, fp16=False,
                         in_nc=3, out_nc=3, clip_size=5, dim=48, 
                         upscale=4, bias=False, num_blocks=6, residual=True, 
                         img_range=255., use_checkpoint=False):
    """
    Convert a TSPANv2 PyTorch model to ONNX format.
    """
    logger.info(f"Loading PyTorch model from: {model_path}")
    device = torch.device('cpu')
    
    # Load model state dict and initialize the model
    state_dict = torch.load(model_path, map_location=device)
    model = load_model_from_state(
        state_dict, in_nc, out_nc, clip_size, 
        dim, upscale, bias, num_blocks, residual, img_range, use_checkpoint
    )
    model.eval()
    model = model.to(device)
    
    logger.info(f"Model Info: clip_size={model.clip_size}, upscale={upscale}x")
    
    # Create the export wrapper around the loaded model
    export_model = TemporalSPANExportWrapper(model)
    export_model.eval()

    # Create dummy input tensor
    dummy_input = torch.randn(*input_shape, dtype=torch.float32, device=device)
    logger.info(f"Using input shape for export: {input_shape}")
    
    dynamic_axes = None
    if dynamic:
        logger.info("Using dynamic axes for batch, height, and width.")
        dynamic_axes = {
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'out_height', 3: 'out_width'}
        }
    
    # Define output paths for FP32 and FP16 models
    base_path = os.path.splitext(onnx_path)[0]
    fp32_path = f"{base_path}_fp32.onnx"

    logger.info(f"\nExporting model to ONNX (FP32): {fp32_path}")
    try:
        torch.onnx.export(
            export_model,              
            dummy_input,               
            fp32_path,
            export_params=True,        
            opset_version=17,          
            do_constant_folding=True,  
            input_names=['input'],     
            output_names=['output'],   
            dynamic_axes=dynamic_axes,
        )
        logger.info(f"Model successfully exported to {fp32_path}")

        if verify:
            verify_onnx_output(export_model, fp32_path, dummy_input)
        
        if fp16:
            fp16_path = f"{base_path}_fp16.onnx"
            export_model_fp16(export_model, dummy_input, fp16_path, dynamic_axes, device)
            
    except Exception as e:
        logger.error(f"❌ Error during ONNX export: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TemporalSPAN model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to the PyTorch model (.pth file)")
    parser.add_argument("--output", type=str, help="Base output path for ONNX model. '_fp32.onnx' and '_fp16.onnx' will be appended.")
    parser.add_argument("--height", type=int, default=256, help="Input height for dummy tensor")
    parser.add_argument("--width", type=int, default=256, help="Input width for dummy tensor")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for dummy tensor")
    parser.add_argument("--clip-size", type=int, default=5, help="Number of frames in input clip (must be odd)")
    parser.add_argument("--in-nc", type=int, default=3, help="Number of input channels")
    parser.add_argument("--out-nc", type=int, default=3, help="Number of output channels")
    parser.add_argument("--dim", type=int, default=48, help="Feature dimension")
    parser.add_argument("--num-blocks", type=int, default=6, help="Number of SPAB blocks per temporal layer")
    parser.add_argument("--upscale", type=int, default=4, help="Upscaling factor")
    parser.add_argument("--bias", action="store_true", help="Use bias in SPAB blocks")
    parser.add_argument("--no-residual", action="store_true", help="Disable residual connection from center frame")
    parser.add_argument("--img-range", type=float, default=255., help="Image range for normalization")
    parser.add_argument("--use-checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic axes for batch, height, and width")
    parser.add_argument("--no-verify", action="store_true", help="Skip ONNX output verification against PyTorch")
    parser.add_argument("--fp16", action="store_true", help="Also create an FP16 version of the model")
    args = parser.parse_args()
    
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.model))[0]
        args.output = f"{base_name}.onnx"
    
    # Define the 4D input shape the script and ONNX model will use
    input_shape = (args.batch, args.clip_size * args.in_nc, args.height, args.width)
    
    convert_model_to_onnx(
        args.model, 
        args.output,
        input_shape,
        args.dynamic,
        not args.no_verify,
        args.fp16,
        args.in_nc,
        args.out_nc,
        args.clip_size,
        args.dim,
        args.upscale,
        args.bias,
        args.num_blocks,
        not args.no_residual,
        args.img_range,
        args.use_checkpoint
    )
