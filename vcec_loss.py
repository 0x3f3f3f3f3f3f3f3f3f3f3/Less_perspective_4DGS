# MIT License
#
# Copyright (c) 2025
#
# Implementation of Velocity-Compensated Edge Consistency (VCEC) Loss
# for Dynamic Scene Reconstruction with 4D Gaussian Splatting
#
# This module implements the VCEC loss function and its simplified variant MER,
# as described in the paper section on "速度补偿的边缘一致性".

import torch
import torch.nn.functional as F
import math


def sobel_edge_extractor(image):
    """
    Extract edges from image using Sobel operator.

    Args:
        image: (C, H, W) tensor, RGB image

    Returns:
        edge_magnitude: (H, W) tensor, edge magnitude
    """
    # Convert to grayscale if RGB
    if image.shape[0] == 3:
        # Use standard RGB to grayscale conversion
        gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        gray = gray.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif image.shape[0] == 9:  # For 9-channel feature maps
        # Average across channels
        gray = image.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H, W)
    else:
        gray = image.unsqueeze(0).unsqueeze(0)

    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)

    # Apply Sobel filters
    edge_x = F.conv2d(gray, sobel_x, padding=1)
    edge_y = F.conv2d(gray, sobel_y, padding=1)

    # Compute edge magnitude
    edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2).squeeze(0).squeeze(0)  # (H, W)

    return edge_magnitude


def compute_pixel_velocity(gaussians, viewpoint_camera, delta_t=1.0):
    """
    Compute pixel-level composite velocity field from 3D Gaussian primitives.

    Implements Equation (1) and (2) from the paper:
    u_i(p) = π(μ_i(t + Δ)) - π(μ_i(t))
    u(p) = Σ_i w_i(p,t) · u_i(p)

    Args:
        gaussians: GaussianModel instance containing primitives
        viewpoint_camera: Camera instance with projection information
        delta_t: Time step for motion computation (default: 1.0 frame)

    Returns:
        velocity_field: (2, H, W) tensor, pixel-level motion field (u_x, u_y)
        weights: (N, H, W) tensor, per-primitive visibility weights
    """
    # Get current positions at time t
    means3D_t = gaussians.get_xyz  # (N, 3)

    # Get velocity from motion parameters
    # In the baseline code, _motion contains polynomial motion coefficients
    # For linear trajectory: means3D(t+Δ) = means3D(t) + v * Δ
    # The first 3 channels of _motion represent linear velocity
    velocity_3d = gaussians._motion[:, 0:3]  # (N, 3)

    # Compute positions at t + delta_t
    means3D_t_plus_delta = means3D_t + velocity_3d * delta_t  # (N, 3)

    # Project both positions to image plane
    # Get camera parameters
    world_view_transform = viewpoint_camera.world_view_transform  # (4, 4)
    proj_matrix = viewpoint_camera.full_proj_transform  # (4, 4)

    # Project means3D_t
    means3D_t_hom = torch.cat([means3D_t, torch.ones_like(means3D_t[:, :1])], dim=1)  # (N, 4)
    means_ndc_t = means3D_t_hom @ proj_matrix.T  # (N, 4)
    means_ndc_t = means_ndc_t / (means_ndc_t[:, 3:4] + 1e-7)  # Perspective division
    means2D_t = means_ndc_t[:, :2]  # (N, 2) in NDC space [-1, 1]

    # Project means3D_t_plus_delta
    means3D_td_hom = torch.cat([means3D_t_plus_delta, torch.ones_like(means3D_t_plus_delta[:, :1])], dim=1)
    means_ndc_td = means3D_td_hom @ proj_matrix.T
    means_ndc_td = means_ndc_td / (means_ndc_td[:, 3:4] + 1e-7)
    means2D_td = means_ndc_td[:, :2]  # (N, 2) in NDC space

    # Compute per-primitive pixel shift: u_i(p) = π(μ_i(t+Δ)) - π(μ_i(t))
    # Convert from NDC to pixel coordinates
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width

    # NDC to pixel: x_pixel = (x_ndc + 1) * W / 2
    pixel_shift = (means2D_td - means2D_t) * torch.tensor([W / 2.0, H / 2.0],
                                                           device=means2D_t.device)  # (N, 2)

    # Note: In practice, we need to integrate this with the rasterizer to get
    # proper visibility weights w_i(p,t). For now, we'll return the shifts
    # and let the calling function handle the weighted composition.

    return pixel_shift, means2D_t, means3D_t


def warp_image_with_flow(image, flow):
    """
    Warp image according to optical flow using bilinear sampling.

    Implements Equation (3):
    Ĩ_{(t+Δ)→t}(p) = I_{t+Δ}(p + u(p))

    Args:
        image: (C, H, W) tensor, image to warp
        flow: (2, H, W) tensor, optical flow field (flow_x, flow_y)

    Returns:
        warped_image: (C, H, W) tensor, warped image
    """
    C, H, W = image.shape

    # Create sampling grid
    # grid_y, grid_x are in range [0, H-1] and [0, W-1]
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=image.device, dtype=torch.float32),
                                     torch.arange(W, device=image.device, dtype=torch.float32),
                                     indexing='ij')

    # Add flow to get sampling locations: p + u(p)
    sample_x = grid_x + flow[0]  # (H, W)
    sample_y = grid_y + flow[1]  # (H, W)

    # Normalize to [-1, 1] for grid_sample
    sample_x = 2.0 * sample_x / (W - 1) - 1.0
    sample_y = 2.0 * sample_y / (H - 1) - 1.0

    # Stack to create grid (H, W, 2)
    grid = torch.stack([sample_x, sample_y], dim=2).unsqueeze(0)  # (1, H, W, 2)

    # Warp image
    warped_image = F.grid_sample(image.unsqueeze(0), grid,
                                  mode='bilinear',
                                  padding_mode='border',
                                  align_corners=True)

    return warped_image.squeeze(0)  # (C, H, W)


def compute_motion_weight_gamma(flow, tau=4.0):
    """
    Compute adaptive motion weight emphasizing fast-moving regions.

    Implements Equation (4):
    γ(p) = clip(||u(p)|| / τ, 0, 1)

    Args:
        flow: (2, H, W) tensor, optical flow field
        tau: Threshold for fast motion (in pixels)

    Returns:
        gamma: (H, W) tensor, motion weights in [0, 1]
    """
    flow_magnitude = torch.norm(flow, p=2, dim=0)  # (H, W)
    gamma = torch.clamp(flow_magnitude / tau, 0.0, 1.0)

    return gamma


def compute_visibility_mask(opacity_t, opacity_td, flow):
    """
    Compute visibility mask based on opacity to handle occlusion.

    Implements Equation (7):
    M(p) = min(A_t(p), A_{t+Δ}(p + u(p)))

    Args:
        opacity_t: (H, W) tensor, opacity at time t
        opacity_td: (H, W) tensor, opacity at time t+delta
        flow: (2, H, W) tensor, optical flow field

    Returns:
        mask: (H, W) tensor, visibility mask
    """
    # Warp opacity_td back to t using flow
    warped_opacity_td = warp_image_with_flow(opacity_td.unsqueeze(0), flow).squeeze(0)

    # Take minimum for visibility mask
    mask = torch.min(opacity_t, warped_opacity_td)

    return mask


def charbonnier_loss(x, epsilon=1e-3):
    """
    Charbonnier loss function: ρ(x) = sqrt(x^2 + ε^2)

    Args:
        x: Input tensor
        epsilon: Small constant for numerical stability

    Returns:
        Charbonnier loss
    """
    return torch.sqrt(x**2 + epsilon**2)


def vcec_loss(render_t, render_td, opacity_t, opacity_td, flow,
              tau=4.0, epsilon=1e-3, lambda_vcec=0.1):
    """
    Compute Velocity-Compensated Edge Consistency (VCEC) loss.

    Implements Equations (5) and (8):
    L_vcec^M = (1/|Ω|) Σ_p M(p)·γ(p)·ρ(G(I_t(p)) - G(Ĩ_{(t+Δ)→t}(p)))

    Args:
        render_t: (C, H, W) tensor, rendered image at time t
        render_td: (C, H, W) tensor, rendered image at time t+delta
        opacity_t: (H, W) tensor, opacity map at time t
        opacity_td: (H, W) tensor, opacity map at time t+delta
        flow: (2, H, W) tensor, pixel-level motion field
        tau: Fast motion threshold (default: 4.0 pixels)
        epsilon: Charbonnier epsilon (default: 1e-3)
        lambda_vcec: Loss weight (default: 0.1)

    Returns:
        loss: Scalar tensor, VCEC loss value
        debug_info: Dictionary with intermediate values for visualization
    """
    # Step 1: Warp render_td back to t using flow (Equation 3)
    warped_render_td = warp_image_with_flow(render_td, flow)

    # Step 2: Extract edges using Sobel operator
    edge_t = sobel_edge_extractor(render_t)  # (H, W)
    edge_warped = sobel_edge_extractor(warped_render_td)  # (H, W)

    # Step 3: Compute motion weight γ(p) (Equation 4)
    gamma = compute_motion_weight_gamma(flow, tau)  # (H, W)

    # Step 4: Compute visibility mask M(p) (Equation 7)
    mask = compute_visibility_mask(opacity_t, opacity_td, flow)  # (H, W)

    # Step 5: Compute edge residual with Charbonnier loss
    edge_residual = edge_t - edge_warped
    residual_loss = charbonnier_loss(edge_residual, epsilon)  # (H, W)

    # Step 6: Apply mask and motion weight
    weighted_loss = mask * gamma * residual_loss  # (H, W)

    # Step 7: Average over all pixels
    loss = weighted_loss.mean() * lambda_vcec

    # Prepare debug info
    debug_info = {
        'warped_render': warped_render_td,
        'edge_t': edge_t,
        'edge_warped': edge_warped,
        'gamma': gamma,
        'mask': mask,
        'flow_magnitude': torch.norm(flow, p=2, dim=0)
    }

    return loss, debug_info


def mer_loss(render_t, gt_image, flow, tau=4.0, epsilon=1e-3, lambda_mer=0.1):
    """
    Motion Edge Reweighting (MER) loss - simplified approximation of VCEC.

    Implements Equation (10):
    L_mer = (1/|Ω|) Σ_p γ(p)·e(p)·ρ(I_t(p) - I_t^gt(p))

    This is a lightweight alternative that doesn't require temporal alignment.

    Args:
        render_t: (C, H, W) tensor, rendered image at time t
        gt_image: (C, H, W) tensor, ground truth image at time t
        flow: (2, H, W) tensor, pixel-level motion field
        tau: Fast motion threshold (default: 4.0 pixels)
        epsilon: Charbonnier epsilon (default: 1e-3)
        lambda_mer: Loss weight (default: 0.1)

    Returns:
        loss: Scalar tensor, MER loss value
        debug_info: Dictionary with intermediate values
    """
    # Step 1: Extract edge magnitude e(p) = ||G(I_t)(p)||
    edge_magnitude = sobel_edge_extractor(render_t)  # (H, W)

    # Step 2: Compute motion weight γ(p)
    gamma = compute_motion_weight_gamma(flow, tau)  # (H, W)

    # Step 3: Compute reconstruction residual
    residual = render_t - gt_image  # (C, H, W)
    residual = residual.pow(2).sum(dim=0).sqrt()  # (H, W), L2 norm across channels
    residual_loss = charbonnier_loss(residual, epsilon)  # (H, W)

    # Step 4: Apply edge and motion weights
    weighted_loss = gamma * edge_magnitude * residual_loss  # (H, W)

    # Step 5: Average over all pixels
    loss = weighted_loss.mean() * lambda_mer

    # Prepare debug info
    debug_info = {
        'edge_magnitude': edge_magnitude,
        'gamma': gamma,
        'flow_magnitude': torch.norm(flow, p=2, dim=0)
    }

    return loss, debug_info


# Helper function to compute flow from rendering outputs
def compute_flow_from_renders(gaussians, viewpoint_cam_t, viewpoint_cam_td,
                               render_func, pipe, background, basicfunction,
                               GRsetting, GRzer, delta_t=1.0):
    """
    Compute optical flow field by rendering at two time steps.

    This is a utility function that interfaces with the rendering pipeline
    to compute the composite velocity field needed for VCEC/MER losses.

    Args:
        gaussians: GaussianModel instance
        viewpoint_cam_t: Camera at time t
        viewpoint_cam_td: Camera at time t+delta
        render_func: Rendering function
        pipe: Pipeline parameters
        background: Background color
        basicfunction: Basis function for temporal interpolation
        GRsetting: Rasterization settings class
        GRzer: Rasterizer class
        delta_t: Time step (default: 1.0)

    Returns:
        flow: (2, H, W) tensor, optical flow field
        render_t: Rendered image at time t
        render_td: Rendered image at time t+delta
        opacity_t: Opacity map at time t
        opacity_td: Opacity map at time t+delta
    """
    # Render at time t
    with torch.no_grad():
        render_pkg_t = render_func(viewpoint_cam_t, gaussians, pipe, background,
                                    basicfunction=basicfunction,
                                    GRsetting=GRsetting, GRzer=GRzer)
        render_t = render_pkg_t["render"]
        depth_t = render_pkg_t["depth"]

        # Get opacity from rendering if available
        if "opacity" in render_pkg_t:
            # Opacity per primitive, need to composite to image
            opacity_t = render_pkg_t["opacity"]
        else:
            # Use depth as proxy for opacity
            opacity_t = (depth_t.squeeze(0) < 15.0).float()

        # Render at time t+delta
        render_pkg_td = render_func(viewpoint_cam_td, gaussians, pipe, background,
                                     basicfunction=basicfunction,
                                     GRsetting=GRsetting, GRzer=GRzer)
        render_td = render_pkg_td["render"]
        depth_td = render_pkg_td["depth"]

        if "opacity" in render_pkg_td:
            opacity_td = render_pkg_td["opacity"]
        else:
            opacity_td = (depth_td.squeeze(0) < 15.0).float()

    # Compute per-primitive pixel shifts
    pixel_shift, means2D_t, means3D_t = compute_pixel_velocity(
        gaussians, viewpoint_cam_t, delta_t
    )

    # For now, create a simple flow field by splatting primitive velocities
    # This is a simplified version - in practice, you'd want to use the full
    # rasterization visibility weights
    H, W = viewpoint_cam_t.image_height, viewpoint_cam_t.image_width
    flow = torch.zeros(2, H, W, device=pixel_shift.device, dtype=pixel_shift.dtype)

    # Convert means2D from NDC to pixel coordinates
    means_pixel_x = ((means2D_t[:, 0] + 1.0) * W / 2.0).long()
    means_pixel_y = ((means2D_t[:, 1] + 1.0) * H / 2.0).long()

    # Clip to image bounds
    valid_mask = (means_pixel_x >= 0) & (means_pixel_x < W) & \
                 (means_pixel_y >= 0) & (means_pixel_y < H)

    means_pixel_x = means_pixel_x[valid_mask]
    means_pixel_y = means_pixel_y[valid_mask]
    valid_shift = pixel_shift[valid_mask]

    # Splat velocities (simple averaging, could be improved with proper splatting)
    flow[0, means_pixel_y, means_pixel_x] = valid_shift[:, 0]
    flow[1, means_pixel_y, means_pixel_x] = valid_shift[:, 1]

    # Apply Gaussian smoothing to create dense flow field
    kernel_size = 5
    sigma = 2.0
    flow = F.gaussian_blur(flow.unsqueeze(0), kernel_size=[kernel_size, kernel_size],
                           sigma=[sigma, sigma]).squeeze(0)

    return flow, render_t, render_td, opacity_t, opacity_td
