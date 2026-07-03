"""Phase 2: FastSAM background segmentation.

Will expose:
    segment_image(image_path) -> PIL.Image   # single-image, used at inference time
    batch_segment(input_dir, output_dir)     # offline batch job (run on Colab)
"""
