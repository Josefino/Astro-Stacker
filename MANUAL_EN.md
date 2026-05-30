# Astro Stacker 2.4 - User Manual

Astro Stacker is an application for stacking astronomical images from DSLR and mirrorless cameras, astronomy cameras, and smart telescopes. It loads a sequence of Light frames, optionally applies Flat/Bias/Dark calibration, aligns the frames on stars or a comet, stacks the result, and provides preview tools for inspection and export.

The program is designed so the linear FIT output remains suitable for further processing. Most controls in the right panel affect the preview and PNG/TIFF export, not the original linear data.

## 1. What Astro Stacker Does

- Stacks Light frames from a selected folder.
- Supports FIT/FITS, common camera RAW files, and standard image formats.
- Applies optional Flat, Bias, and Dark calibration.
- Aligns frames by translation, affine ECC, stars with RANSAC, perspective star matching, or comet motion.
- Allows automatic or manual reference frame selection.
- Filters the best frames by quality score.
- Shows a diagnostic frame quality table.
- Saves and loads settings profiles.
- Provides preview tools with histogram, curves, color controls, background balancing, and Balance.
- Exports linear FIT or visually adjusted PNG/TIFF.
- Can launch the PixInsight wrapper.

## 2. Quick Start for Normal Stacking

1. Click **Choose folder** and select a folder with Light frames.
2. Enable **RAW only** if the folder also contains JPG/PNG/BMP/TIFF preview files.
3. For a normal deep-sky stack, keep these settings:
   - **Alignment:** Star alignment - stars + RANSAC
   - **Stacking:** Sigma-clipped mean
   - **Automatically choose best reference:** enabled
   - **Use only best frames:** enabled as needed
4. Click **Start stacking**.
5. After stacking, check the preview, diagnostic table, and the used/excluded frame summary.
6. Save the result with **File > Save result as**.

## 3. Left Panel - Stacking Controls

### Folder and Preview Selection

**Choose folder** loads the working folder. The file list shows Light, Flat, Bias, and Dark frames. After stacking, the list marks:

- `*` the reference frame,
- `x` excluded frames.

The file list can also be used to inspect individual frames.

### RAW Only

When enabled, **RAW only** keeps FIT/FITS and camera RAW files such as ARW, CR2, CR3, and similar formats. It excludes JPG, PNG, BMP, and TIFF files. This is useful when a camera or smart telescope stores both working frames and preview images in the same folder.

### Alignment Modes

**Translation**  
Fast shift-only alignment. Suitable for small drift without rotation.

**Calibration/no alignment**  
Stacks calibration frames pixel-to-pixel.

**ECC affine**  
Uses image correlation to estimate shift, rotation, and scale. It can be slower and may not work well on weak objects.

**Star alignment + RANSAC**  
The recommended main mode. The program detects stars, matches them against the reference frame, and estimates the transformation with RANSAC. Frames without a valid alignment are rejected.

**Star perspective - stars + corners**  
An experimental mode for more complex distortion. Use it only when normal RANSAC alignment is not enough.

**Comet alignment**  
Stacks on the comet position.

**Star + Comet**  
Creates separate star and comet stack outputs.

### Automatic and Manual Reference

**Automatically choose best reference** selects the frame with the best quality score. The score is based mainly on sharpness and detected star count.

If automatic reference selection does not work well for a specific sequence, select a suitable frame in the file list and click **Use current frame as reference**. This disables automatic reference selection and uses the selected frame instead.

### Quality Filter

The program computes a quality score for each Light frame. The score combines:

- frame sharpness,
- detected star count.

**Keep** defines what percentage of the best frames will be used. For example, 80% keeps the best 80% and excludes the worst 20% before alignment.

Important: a frame can pass the quality filter and still be rejected later if alignment fails.

### Max. Star Drift

Defines the maximum expected star displacement relative to the reference frame. Normal sequences can use a smaller value. EAA, dithering, or smart telescope sequences may need a larger value.

### Ignore Border

Ignores the image borders during star detection. This helps when the edges contain branches, noise, vignetting, black corners after rotation, or other distracting structures.

### Strict Star Filter

Applies stricter star shape filtering. It helps suppress branches, trails, and elongated artifacts. If too many usable frames are rejected, disabling this option can sometimes help.

## 4. Diagnostic Frame Quality Table

The lower part of the window contains the **Frame quality** table. It helps explain why frames were used or excluded.

Columns:

- `#` frame order,
- `File` file name,
- `Score` overall quality score,
- `Stars` detected star count,
- `Sharpness` Laplacian variance sharpness metric,
- `Status` frame status,
- `Ref` reference marker.

Typical statuses:

- **Used** - the frame was stacked.
- **Reference** - the reference frame.
- **Excluded by quality** - the frame failed the quality filter.
- **Rejected alignment** - the frame passed quality filtering, but alignment failed.
- **Skipped** - the frame was not part of the stack, for example a calibration or unused file type.

Clicking a row switches the preview to that frame.

### Reviewing Frames Before Stacking

**Review frames before stacking** adds a manual review step before the actual stack begins.

Workflow:

1. Enable **Review frames before stacking**.
2. Click **Start stacking**.
3. The program only computes frame quality, chooses the reference, and fills the Frame Quality table.
4. Select a frame in the table that you do not want to use.
5. Press Space. The status changes to **Excluded** and the row turns red.
6. Press Space again to allow the frame again.
7. Click **Continue stacking**.

The reference frame cannot be excluded with Space. If you want to exclude the reference, choose another reference frame first.

Manual exclusion has priority over the automatic selection. The quality filter still makes the initial selection, but before final alignment you can manually remove additional frames.

## 5. Flat, Bias, and Dark Calibration

Calibration frames can be used in two ways:

- automatically, if they are in appropriately named folders,
- manually with the **Flat**, **Bias**, and **Dark** buttons in the right panel.

Calibration is applied before alignment and stacking.

**Reset calibration** removes the selected calibration frames.

## 6. Comet Stacking

For comet stacking, the most reliable method is to mark the comet position in two frames.

1. Select the folder with the sequence.
2. Click **Comet First**.
3. The program loads the first frame. Click the comet nucleus in the preview.
4. Click **Comet Last**.
5. The program loads the last frame. Click the comet nucleus again.
6. The program now knows the comet motion between the beginning and end of the sequence.
7. Select **Comet alignment** or **Star + Comet**.
8. Start stacking.

**Comet Clear** removes both saved comet positions.

Comet options:

- **Max. comet motion** - maximum expected comet movement.
- **Refine comet position** - refines the nucleus position with local correlation.
- **Comet template** - template size around the nucleus.
- **Comet search** - how far from the predicted position the nucleus may be searched.

## 7. Settings Profiles

Settings profiles are saved as JSON files.

Use:

- **File > Save settings profile**
- **File > Load settings profile**

A profile stores stacking, calibration, comet, and preview settings. Profiles are useful for different workflows, such as:

- DSLR RAW,
- Seestar/Vespera EAA,
- comet processing,
- calibration frame stacking,
- linear FIT workflows.

## 8. Right Panel - Preview and Adjustments

The right panel is used for inspection and visual adjustments. These controls mainly affect preview and PNG/TIFF export. Linear FIT data remains suitable for further processing.

### Show Stacked Image

Restores the preview to the original stacked image without crop, neutralization, or visual adjustments.

### Balance

**Balance** automatically sets a reasonable preview for a linear image:

- neutralizes the background for preview,
- sets the black point,
- sets gamma,
- makes weak linear data easier to inspect.

Balance does not modify the linear FIT data. It is a preview helper.

### Black Point, White Point, Gamma

Basic display curve controls:

- **Black point** moves the black level.
- **White point** sets the bright level.
- **Gamma** adjusts midtones.

### Highlight Compression

Compresses very bright highlights, such as bright stars or object cores. Useful for previewing high dynamic range images.

### Vignette Removal

Gently brightens edges and corners. It is not a replacement for real Flat calibration, but it can help with quick visual inspection.

### Synthetic Flat

Estimates a smooth background from the stacked image and uses it as a gentle synthetic flat. Use carefully, especially with large nebulae, where the program may mistake real nebulosity for background.

### Color Background Correction

Suppresses smooth color casts in the background per RGB channel. This is useful for strongly unbalanced images, such as pink or purple backgrounds from smart telescopes.

### SCNR Green

Suppresses green cast. Use it only when green is visibly problematic.

### Contrast, Saturation, RGB

- **Contrast** adjusts preview contrast.
- **Saturation** adjusts color intensity.
- **Red, Green, Blue** manually adjust individual channels.

### Histogram

The histogram shows luminance and RGB channels. It helps verify whether data is clipped in shadows or highlights.

### Crop Edges

Crops the selected percentage from each edge of the current image. Useful after rotation, dithering, or EAA sequences with dark corners.

### Auto White Balance

Attempts automatic white balance on neutral/gray regions. For some astro images, manual RGB adjustment or Color background correction may work better.

### Neutralize Background

Attempts to equalize the background color. It works best when there is enough neutral background without nebulosity and without problematic edges.

### Flip Horizontal / Vertical

Flips the preview horizontally or vertically.

### Fit and 1:1

- **Fit** fits the image to the window.
- **1:1** shows real pixels.

## 9. PixInsight Wrapper

The **PixInsight** button opens PixInsight and runs the AS_Stacker wrapper if it is available. The wrapper can run the CLI stacker from PixInsight and open the resulting FIT file after completion.

The wrapper needs a Python environment with the required packages installed.

## 10. Recommended Workflows

### Normal Deep-Sky Sequence

- Star alignment + RANSAC
- Sigma-clipped mean
- Auto reference enabled
- Quality filter 80-100%
- Strict star filter enabled

### Large Drift or Dithering

- Increase Max. star drift.
- Try a manual reference from the middle of the sequence.
- Use the diagnostic table to see whether frames fail quality filtering or alignment.

### Smart Telescopes and EAA

- Enable RAW only if preview images are mixed with stackable frames.
- Use Crop edges for rotation and dark corners.
- Use Color background correction for pink or purple backgrounds.

### Comets

- Mark both first and last comet positions.
- Use Comet alignment or Star + Comet.
- Keep comet refinement enabled for weak nuclei.

## 11. Troubleshooting

### Too Many Frames Are Rejected

Check the Frame quality table:

- if frames are **Excluded by quality**, reduce quality filtering or increase Keep %;
- if frames are **Rejected alignment**, increase Max. star drift, change the reference frame, or try another alignment mode.

### The Image Has Strange Colors

Try, in order:

1. Balance,
2. Color background correction,
3. Neutralize background,
4. manual RGB adjustments,
5. crop the edges and neutralize again.

### The Result Has Dark Corners

This is common with rotation, dithering, or large drift. Use Crop edges or crop the result later in another editor.

### FIT Output Looks Dark

This is normal. FIT output is linear and needs stretching. Use Balance for preview, or process the output in PixInsight, Siril, or another astrophotography editor.
