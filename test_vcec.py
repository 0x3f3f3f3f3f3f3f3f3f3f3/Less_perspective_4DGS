#!/usr/bin/env python3
"""
Simple test script to verify VCEC implementation.
This tests basic functionality without requiring full training setup.
"""

import torch
import torch.nn.functional as F
from vcec_loss import (
    sobel_edge_extractor,
    compute_motion_weight_gamma,
    warp_image_with_flow,
    compute_visibility_mask,
    vcec_loss,
    mer_loss,
    charbonnier_loss
)


def test_sobel_edge_extractor():
    """Test Sobel edge extraction."""
    print("Testing Sobel edge extractor...")

    # Create a simple test image with an edge
    image = torch.zeros(3, 64, 64)
    image[:, :, 32:] = 1.0  # Vertical edge in the middle

    edges = sobel_edge_extractor(image)

    assert edges.shape == (64, 64), f"Expected shape (64, 64), got {edges.shape}"
    assert edges.max() > 0, "Edge detector should detect something"

    # The edge should be strongest around column 32
    max_col = edges.sum(dim=0).argmax().item()
    assert 30 <= max_col <= 34, f"Edge should be near column 32, got {max_col}"

    print("✓ Sobel edge extractor works correctly")


def test_motion_weight_gamma():
    """Test motion weight computation."""
    print("Testing motion weight gamma...")

    # Create flow field with varying magnitudes
    flow = torch.zeros(2, 64, 64)
    flow[0, :32, :] = 2.0  # Slow motion region (magnitude 2)
    flow[0, 32:, :] = 8.0  # Fast motion region (magnitude 8)

    gamma = compute_motion_weight_gamma(flow, tau=4.0)

    assert gamma.shape == (64, 64), f"Expected shape (64, 64), got {gamma.shape}"
    assert gamma.min() >= 0 and gamma.max() <= 1, "Gamma should be in [0, 1]"

    # Slow region should have lower weight
    assert gamma[:32].mean() < gamma[32:].mean(), "Fast motion should have higher weight"

    print("✓ Motion weight gamma works correctly")


def test_warp_image_with_flow():
    """Test image warping with optical flow."""
    print("Testing image warping...")

    # Create test image
    image = torch.zeros(3, 64, 64)
    image[:, 20:40, 20:40] = 1.0  # White square in center

    # Create flow that shifts right by 10 pixels
    flow = torch.zeros(2, 64, 64)
    flow[0, :, :] = 10.0  # Shift right

    warped = warp_image_with_flow(image, flow)

    assert warped.shape == image.shape, f"Warped image shape mismatch: {warped.shape} vs {image.shape}"

    # After warping, the square should be shifted
    # Original square is at 20:40, after +10 shift should be at 30:50
    assert warped[:, 20:40, 30:50].mean() > 0.5, "Warped square should be visible at new location"

    print("✓ Image warping works correctly")


def test_visibility_mask():
    """Test visibility mask computation."""
    print("Testing visibility mask...")

    opacity_t = torch.ones(64, 64) * 0.8
    opacity_td = torch.ones(64, 64) * 0.9
    flow = torch.zeros(2, 64, 64)

    mask = compute_visibility_mask(opacity_t, opacity_td, flow)

    assert mask.shape == (64, 64), f"Expected shape (64, 64), got {mask.shape}"
    assert torch.allclose(mask, torch.ones(64, 64) * 0.8, atol=0.01), \
        "Mask should be minimum of opacities"

    print("✓ Visibility mask works correctly")


def test_charbonnier_loss():
    """Test Charbonnier loss function."""
    print("Testing Charbonnier loss...")

    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    loss = charbonnier_loss(x, epsilon=1e-3)

    # Charbonnier should be close to |x| for large x
    assert torch.allclose(loss[3], torch.tensor(3.0), atol=0.01), \
        "Charbonnier should approximate abs for large values"

    # For x=0, should equal epsilon
    assert torch.allclose(loss[0], torch.tensor(1e-3), atol=1e-4), \
        "Charbonnier(0) should equal epsilon"

    print("✓ Charbonnier loss works correctly")


def test_mer_loss():
    """Test MER loss computation."""
    print("Testing MER loss...")

    # Create test images
    render_t = torch.rand(3, 64, 64)
    gt_image = render_t + torch.randn(3, 64, 64) * 0.1  # Add noise

    # Create flow
    flow = torch.ones(2, 64, 64) * 2.0

    loss, debug_info = mer_loss(render_t, gt_image, flow, tau=4.0, lambda_mer=0.1)

    assert isinstance(loss.item(), float), "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    assert 'edge_magnitude' in debug_info, "Debug info should contain edge_magnitude"
    assert 'gamma' in debug_info, "Debug info should contain gamma"

    print(f"✓ MER loss works correctly (loss value: {loss.item():.4f})")


def test_vcec_loss():
    """Test VCEC loss computation."""
    print("Testing VCEC loss...")

    # Create test images
    render_t = torch.rand(3, 64, 64)
    render_td = torch.rand(3, 64, 64)

    # Create opacity maps
    opacity_t = torch.rand(64, 64) * 0.5 + 0.5  # [0.5, 1.0]
    opacity_td = torch.rand(64, 64) * 0.5 + 0.5

    # Create flow
    flow = torch.ones(2, 64, 64) * 3.0

    loss, debug_info = vcec_loss(
        render_t, render_td, opacity_t, opacity_td, flow,
        tau=4.0, lambda_vcec=0.1
    )

    assert isinstance(loss.item(), float), "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    assert 'warped_render' in debug_info, "Debug info should contain warped_render"
    assert 'edge_t' in debug_info, "Debug info should contain edge_t"
    assert 'mask' in debug_info, "Debug info should contain mask"

    print(f"✓ VCEC loss works correctly (loss value: {loss.item():.4f})")


def test_integration():
    """Test that VCEC can be used in a training-like scenario."""
    print("Testing integration scenario...")

    # Simulate a mini training iteration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    # Create dummy data on device
    render_t = torch.rand(3, 128, 128, device=device)
    render_td = torch.rand(3, 128, 128, device=device)
    gt_image = render_t.clone() + torch.randn(3, 128, 128, device=device) * 0.05

    opacity_t = torch.ones(128, 128, device=device) * 0.9
    opacity_td = torch.ones(128, 128, device=device) * 0.85

    flow = torch.randn(2, 128, 128, device=device) * 2.0

    # Test MER loss
    loss_mer, _ = mer_loss(render_t, gt_image, flow, lambda_mer=0.05)
    loss_mer.backward()

    # Test VCEC loss
    render_t = torch.rand(3, 128, 128, device=device, requires_grad=True)
    loss_vcec, _ = vcec_loss(render_t, render_td, opacity_t, opacity_td, flow, lambda_vcec=0.1)
    loss_vcec.backward()

    assert render_t.grad is not None, "Gradients should be computed"
    assert not torch.isnan(render_t.grad).any(), "Gradients should not contain NaN"

    print("✓ Integration test passed (gradients flow correctly)")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("VCEC Implementation Test Suite")
    print("="*60)

    tests = [
        test_sobel_edge_extractor,
        test_motion_weight_gamma,
        test_warp_image_with_flow,
        test_visibility_mask,
        test_charbonnier_loss,
        test_mer_loss,
        test_vcec_loss,
        test_integration,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
        print()

    print("="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_all_tests())
