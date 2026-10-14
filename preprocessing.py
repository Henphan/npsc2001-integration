#!/usr/bin/env python
# coding: utf-8

# Fiber Enhancement & Mask Generation - by Ferdinando Giuliano

# In[ ]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util, morphology, measure
from skimage.filters import frangi, gabor, gaussian, threshold_multiotsu
from scipy import ndimage
from skimage.morphology import skeletonize
from tqdm import tqdm
from pathlib import Path

# In[2]:


class RNFLImageProcessor:
    """Process RNFL images: crop, enhance, detect fibers and vessels"""
    
    def __init__(self, crop_params=None):
        """
        Initialize processor with crop parameters
        
        Args:
            crop_params: dict with keys 'top', 'bottom', 'left', 'right'
                        If None, will use adaptive cropping per image
        """
        self.crop_params = crop_params
        self.use_adaptive_crop = (crop_params is None)
    
    def crop_image(self, image):
            """
            Apply adaptive top/bottom crop to image or mask
            
            Args:
                image: Image to crop (2D or 3D array)
                
            Returns:
                Cropped image and crop parameters used
            """
            if not self.use_adaptive_crop and self.crop_params is not None:
                # Use fixed crop parameters
                crop_top = self.crop_params['top']
                crop_bottom = self.crop_params['bottom']
            else:
                # Adaptive cropping based on image height
                h = image.shape[0]
                
                # FIXED TOP CROP: Always 100 pixels
                crop_top = 100
                
                # HEIGHT-BASED BOTTOM CROP RULE
                # If height < 1000: crop bottom 110 pixels
                # If height >= 1000: crop bottom 220 pixels
                crop_bottom = 110 if h < 1000 else 220
            
            # Apply crop
            h = image.shape[0]
            cropped = image[crop_top:h-crop_bottom]
            
            # Return cropped image and parameters used
            crop_params = {'top': crop_top, 'bottom': crop_bottom}
            return cropped, crop_params
    
    def refine_vessel_mask(self, vessel_mask):
        """
        Refine vessel mask: remove noise, fill holes, smooth boundaries
        
        Args:
            vessel_mask: Binary vessel mask (numpy array)
            
        Returns:
            vessel_mask_refined, vessel_mask_dilated
        """
        # Ensure binary
        vessel_mask_original = vessel_mask > 0.5
        
        # Remove small noise
        vessel_mask_cleaned = morphology.remove_small_objects(
            vessel_mask_original, 
            min_size=50
        )
        
        # Fill small holes
        vessel_mask_filled = morphology.remove_small_holes(
            vessel_mask_cleaned, 
            area_threshold=100
        )
        
        # Smooth boundaries
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vessel_mask_smoothed = cv2.morphologyEx(
            vessel_mask_filled.astype(np.uint8), 
            cv2.MORPH_CLOSE, 
            smooth_kernel, 
            iterations=2
        )
        
        # Keep only vessel-like structures (elongated)
        labeled_vessels = measure.label(vessel_mask_smoothed)
        regions_vessels = measure.regionprops(labeled_vessels)
        
        vessel_mask_refined = np.zeros_like(vessel_mask_smoothed, dtype=bool)
        for region in regions_vessels:
            if region.eccentricity > 0.7 or region.area > 200:
                vessel_mask_refined[labeled_vessels == region.label] = True
        
        # Dilate for safety margin
        vessel_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        vessel_mask_dilated = cv2.dilate(
            vessel_mask_refined.astype(np.uint8), 
            vessel_kernel, 
            iterations=1
        )
        
        return vessel_mask_refined, vessel_mask_dilated
    
    def detect_fibers(self, img_gray_u8, vessel_mask_dilated):
        """
        Detect fibers using Frangi + Gabor filters
        
        Args:
            img_gray_u8: Grayscale image (uint8)
            vessel_mask_dilated: Dilated vessel mask to exclude
            
        Returns:
            fiber_mask_final: Binary fiber mask
        """
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray_u8)
        
        # Remove vessels
        background_intensity = np.median(img_clahe[vessel_mask_dilated == 0])
        img_no_vessels = img_clahe.copy()
        img_no_vessels[vessel_mask_dilated > 0] = background_intensity
        
        # Smooth
        fibers_smooth = gaussian(img_no_vessels, sigma=0.8)
        
        # Frangi filter
        img_frangi_fibers = frangi(
            fibers_smooth, 
            sigmas=range(1, 5, 1),
            black_ridges=False,
            alpha=0.5,
            beta=0.5,
            gamma=20
        )
        
        # Gabor filter bank
        gabor_response = np.zeros_like(fibers_smooth, dtype=float)
        orientations = np.arange(0, np.pi, np.pi / 8)
        gabor_frequency = 0.15
        
        for theta in orientations:
            filt_real, filt_imag = gabor(
                fibers_smooth, 
                frequency=gabor_frequency,
                theta=theta,
                sigma_x=2,
                sigma_y=2
            )
            gabor_magnitude = np.sqrt(filt_real**2 + filt_imag**2)
            np.maximum(gabor_response, gabor_magnitude, out=gabor_response)
        
        # Combine responses
        frangi_norm = (img_frangi_fibers - img_frangi_fibers.min()) / (img_frangi_fibers.max() - img_frangi_fibers.min() + 1e-8)
        gabor_norm = (gabor_response - gabor_response.min()) / (gabor_response.max() - gabor_response.min() + 1e-8)
        combined_response = 0.7 * frangi_norm + 0.3 * gabor_norm
        
        # Multi-Otsu thresholding
        thresholds = threshold_multiotsu(combined_response, classes=3)
        mask_all_fibers = combined_response > thresholds[0]
        
        # Morphological refinement
        mask_cleaned = morphology.remove_small_objects(mask_all_fibers, min_size=10)
        mask_cleaned = morphology.remove_small_holes(mask_cleaned, area_threshold=30)
        
        # Shape-based filtering
        labeled = measure.label(mask_cleaned)
        regions = measure.regionprops(labeled)
        
        fiber_mask_final = np.zeros_like(mask_cleaned, dtype=bool)
        for region in regions:
            if region.area > 10:
                if region.eccentricity > 0.6 or region.area > 50:
                    fiber_mask_final[labeled == region.label] = True
        
        # Remove vessel contamination
        fiber_mask_final[vessel_mask_dilated > 0] = False
        
        return fiber_mask_final
    
    def process_single_image(self, image_path, vessel_mask_path, output_dir_images, output_dir_masks):
        """
        Process a single image: crop, refine vessel mask, detect fibers
        
        Args:
            image_path: Path to input image
            vessel_mask_path: Path to vessel mask
            output_dir_images: Output directory for cropped images
            output_dir_masks: Output directory for masks
            
        Returns:
            dict with paths to saved files and processing stats
        """
        # Load image
        img_color = io.imread(image_path)
        if img_color.ndim == 3:
            img_gray = color.rgb2gray(img_color)
        else:
            img_gray = img_color

        # Crop image (automatically detects bounds if adaptive)
        img_gray_cropped, crop_params = self.crop_image(img_gray)
        img_gray_u8 = util.img_as_ubyte(img_gray_cropped)
        raise Exception("asdf")
        
        # Load and crop vessel mask with SAME parameters
        vessel_mask = io.imread(vessel_mask_path)
        if vessel_mask.ndim == 3:
            vessel_mask = color.rgb2gray(vessel_mask)
        
        # Temporarily override crop params to use the same ones
        saved_use_adaptive = self.use_adaptive_crop
        saved_crop_params = self.crop_params
        
        self.use_adaptive_crop = False
        self.crop_params = crop_params
        vessel_mask_cropped, _ = self.crop_image(vessel_mask)
        
        # Restore original settings
        self.use_adaptive_crop = saved_use_adaptive
        self.crop_params = saved_crop_params
        
        # Refine vessel mask
        vessel_mask_refined, vessel_mask_dilated = self.refine_vessel_mask(vessel_mask_cropped)
        
        # Takes in vessel_mask_refined instead of vessel_mask_dilated
        # Detect fibers
        fiber_mask_final = self.detect_fibers(img_gray_u8, vessel_mask_refined)
        
        # Save results
        base_name = Path(image_path).stem
        
        # Save cropped image
        image_out_path = os.path.join(output_dir_images, f"{base_name}_new.png")
        cv2.imwrite(image_out_path, img_gray_u8)
        
        # Save refined vessel mask
        vessel_out_path = os.path.join(output_dir_masks, f"{base_name}_vessel_new.png")
        # Take in vessel_mask_refined instead of vessel_mask_dilated
        cv2.imwrite(vessel_out_path, (vessel_mask_refined * 255).astype(np.uint8))
        
        # Save fiber mask
        fiber_out_path = os.path.join(output_dir_masks, f"{base_name}_fiber_new.png")
        cv2.imwrite(fiber_out_path, (fiber_mask_final * 255).astype(np.uint8))
        
        # Calculate stats
        stats = {
            'image_path': image_out_path,
            'vessel_mask_path': vessel_out_path,
            'fiber_mask_path': fiber_out_path,
            'image_size': img_gray_cropped.shape,
            'crop_params': crop_params,
            'vessel_pixels': int(np.sum(vessel_mask_dilated)),
            'fiber_pixels': int(np.sum(fiber_mask_final)),
            'fiber_coverage': float(100 * np.sum(fiber_mask_final) / img_gray_cropped.size)
        }
        
        return stats

# In[3]:


def find_image_mask_pairs(images_dir, masks_dir):
    """
    Find all image-mask pairs in the dataset
    
    Returns:
        list of dicts with 'image', 'vessel_mask', 'base_name'
    """
    pairs = []
    
    # Find all images
    image_files = sorted([f for f in os.listdir(images_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        
        # Find corresponding vessel mask
        vessel_mask_candidates = [
            f"{base_name}_vessel.png",
            f"{base_name} _vessel.png",  # Handle potential space
        ]
        
        vessel_mask_path = None
        for candidate in vessel_mask_candidates:
            candidate_path = os.path.join(masks_dir, candidate)
            if os.path.exists(candidate_path):
                vessel_mask_path = candidate_path
                break
        
        if vessel_mask_path:
            pairs.append({
                'image': os.path.join(images_dir, img_file),
                'vessel_mask': vessel_mask_path,
                'base_name': base_name
            })
        else:
            print(f"Warning: No vessel mask found for {img_file}")
    
    return pairs

# In[4]:


def visualize_results(pairs, output_dir_images, output_dir_masks, num_examples=5):
    """
    Visualize first N examples with images and masks
    
    Args:
        pairs: List of processed image-mask pairs
        output_dir_images: Directory with cropped images
        output_dir_masks: Directory with masks
        num_examples: Number of examples to show
    """
    num_examples = min(num_examples, len(pairs))
    
    fig, axes = plt.subplots(num_examples, 4, figsize=(20, 5*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i, pair in enumerate(pairs[:num_examples]):
        base_name = pair['base_name']
        
        # Load processed image
        img_path = os.path.join(output_dir_images, f"{base_name}_new.png")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Load vessel mask
        vessel_path = os.path.join(output_dir_masks, f"{base_name}_vessel_new.png")
        vessel_mask = cv2.imread(vessel_path, cv2.IMREAD_GRAYSCALE)
        
        # Load fiber mask
        fiber_path = os.path.join(output_dir_masks, f"{base_name}_fiber_new.png")
        fiber_mask = cv2.imread(fiber_path, cv2.IMREAD_GRAYSCALE)
        
        # Original image
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'{base_name}\nOriginal (Cropped)')
        axes[i, 0].axis('off')
        
        # Vessel mask overlay
        overlay_vessel = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        overlay_vessel[vessel_mask > 0] = [255, 0, 0]  # Red for vessels
        axes[i, 1].imshow(overlay_vessel)
        axes[i, 1].set_title('Vessel Mask\n(Red = Vessels)')
        axes[i, 1].axis('off')
        
        # Fiber mask overlay
        overlay_fiber = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        overlay_fiber[fiber_mask > 0] = [0, 255, 0]  # Green for fibers
        axes[i, 2].imshow(overlay_fiber)
        axes[i, 2].set_title('Fiber Mask\n(Green = Fibers)')
        axes[i, 2].axis('off')
        
        # Combined overlay
        overlay_combined = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        overlay_combined[vessel_mask > 0] = [255, 0, 0]  # Red for vessels
        overlay_combined[fiber_mask > 0] = [0, 255, 0]   # Green for fibers
        axes[i, 3].imshow(overlay_combined)
        axes[i, 3].set_title('Combined\n(Red=Vessels, Green=Fibers)')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_masks, 'visualization_examples.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "-"*80)
    print("MASK COLOR LEGEND:")
    print("-"*80)
    print("RED = Vessels (blood vessels)")
    print("GREEN = Fibers (RNFL nerve fibers)")
    print("BLACK = Background")
    print("-"*80)

# In[7]:


def main_process():
    """Main processing pipeline"""
    
    # Configuration
    images_dir = "images"
    masks_dir = "masks"
    output_dir_images = "new_images"
    output_dir_masks = "new_masks"
    
    # Crop parameters
    crop_params = None  # ADAPTIVE MODE
    
    # Create output directories
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_masks, exist_ok=True)
    
    print("\n" + "-"*80)
    print("RNFL IMAGE PROCESSING PIPELINE")
    print("-"*80)
    print(f"Input images:       {images_dir}")
    print(f"Input masks:        {masks_dir}")
    print(f"Output images:      {output_dir_images}")
    print(f"Output masks:       {output_dir_masks}")
    print(f"Crop mode:          {'ADAPTIVE (auto-detect per image)' if crop_params is None else f'FIXED {crop_params}'}")
    print("-"*80 + "\n")
    
    # Find all image-mask pairs
    print("Finding image-mask pairs...")
    pairs = find_image_mask_pairs(images_dir, masks_dir)
    print(f"Found {len(pairs)} image-mask pairs\n")
    
    if len(pairs) == 0:
        print("No valid image-mask pairs found!")
        return
    
    # Initialize processor
    processor = RNFLImageProcessor(crop_params=crop_params)
    
    # Process all images
    print("Processing images...")
    all_stats = []
    
    for pair in tqdm(pairs, desc="Processing"):
        try:
            stats = processor.process_single_image(
                pair['image'],
                pair['vessel_mask'],
                output_dir_images,
                output_dir_masks
            )
            stats['base_name'] = pair['base_name']
            all_stats.append(stats)
            
        except Exception as e:
            print(f"\nError processing {pair['base_name']}: {e}")
            continue
    
    print(f"\nSuccessfully processed {len(all_stats)} images\n")
    
    # Print summary statistics
    print("-"*80)
    print("PROCESSING SUMMARY (First 10 images)")
    print("-"*80)
    for stats in all_stats[:10]:
        print(f"\n{stats['base_name']}:")
        print(f"  Size:            {stats['image_size']}")
        if crop_params is None:
            print(f"  Crop applied:    top={stats['crop_params']['top']}, bottom={stats['crop_params']['bottom']}")
        print(f"  Vessel pixels:   {stats['vessel_pixels']:,}")
        print(f"  Fiber pixels:    {stats['fiber_pixels']:,}")
        print(f"  Fiber coverage:  {stats['fiber_coverage']:.2f}%")
    
    if len(all_stats) > 10:
        print(f"\n... and {len(all_stats) - 10} more images")
    
    # Calculate overall statistics
    total_fiber_pixels = sum(s['fiber_pixels'] for s in all_stats)
    total_vessel_pixels = sum(s['vessel_pixels'] for s in all_stats)
    avg_fiber_coverage = np.mean([s['fiber_coverage'] for s in all_stats])
    
    print("\n" + "-"*80)
    print("OVERALL STATISTICS")
    print("-"*80)
    print(f"Total images processed:  {len(all_stats)}")
    print(f"Total vessel pixels:     {total_vessel_pixels:,}")
    print(f"Total fiber pixels:      {total_fiber_pixels:,}")
    print(f"Average fiber coverage:  {avg_fiber_coverage:.2f}%")
    print("-"*80 + "\n")
    
    # Visualize first 5 examples
    print("Generating visualizations...")
    visualize_results(pairs, output_dir_images, output_dir_masks, num_examples=5)
    
    print("\nProcessing complete!")
    print(f"Cropped images saved to: {output_dir_images}/")
    print(f"Masks saved to: {output_dir_masks}/")
    print(f"Visualization saved to: {output_dir_masks}/visualization_examples.png")

# In[ ]:


main_process()
