from src.distribution.wrapped_normal import WrappedNormalLorentz
import torch
from src.manifolds.lorentz import Lorentz

def test_wrapped_normal_lorentz():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize manifold
    curvature = 1.0
    manifold = Lorentz(k=curvature)
    print(f"Created Lorentz manifold with curvature {curvature}")
    
    # Create parameters
    batch_size = 3
    dim = 4  # Ambient dimension for Lorentz
    mu = torch.randn((batch_size, dim-1), device=device) * 0.1
    log_var = torch.randn((batch_size, dim-1), device=device) - 2  # Small variance
    print(f"Created parameters: mu shape {mu.shape}, log_var shape {log_var.shape}")
    
    # Initialize distribution
    try:
        dist = WrappedNormalLorentz(mu, log_var, manifold)
        print("Successfully initialized WrappedNormalLorentz")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return
    
    # Test mean property
    try:
        mean = dist.mean
        print(f"Mean shape: {mean.shape}")
        print(f"Mean equals mu: {torch.allclose(mean, mu)}")
    except Exception as e:
        print(f"Error accessing mean: {e}")
    
    # Test scale property
    try:
        scale = dist.scale
        print(f"Scale shape: {scale.shape}")
        expected_scale = torch.exp(0.5 * log_var)
        print(f"Scale calculated correctly: {torch.allclose(scale, expected_scale)}")
    except Exception as e:
        print(f"Error accessing scale: {e}")
    
    # Test sampling
    try:
        n_samples = 5
        samples = dist.rsample((n_samples,))
        print(f"Generated {n_samples} samples with shape {samples.shape}")
        
        # Check if samples are on the manifold
        on_manifold = manifold.check_point_on_manifold(samples)
        print(f"All samples on manifold: {on_manifold}")
    except Exception as e:
        print(f"Sampling error: {e}")
    
    # Test log_prob
    try:
        log_probs = dist.log_prob(samples)
        print(f"Log probabilities shape: {log_probs.shape}")
        print(f"Log probabilities finite: {torch.isfinite(log_probs).all().item()}")
        print(f"Log probability range: [{log_probs.min().item():.4f}, {log_probs.max().item():.4f}]")
    except Exception as e:
        print(f"Log probability error: {e}")

if __name__ == "__main__":
    test_wrapped_normal_lorentz()