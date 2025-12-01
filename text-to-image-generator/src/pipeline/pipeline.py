"""
End-to-end generative AI pipeline
"""

from .generator import TextToImageGenerator


class GenerativeAIPipeline:
    """
    Complete automated pipeline for text-to-image generation

    Pipeline Steps:
    1. Text Input Processing
    2. Image Generation with Attention
    3. Quality Evaluation
    4. Result Saving
    """
    
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5"):
        print("\n" + "="*70)
        print("🔄 INITIALIZING GENERATIVE AI PIPELINE")
        print("="*70 + "\n")

        self.system = TextToImageGenerator(mode="pretrained", model_name=model_name)

        print("✅ Pipeline ready for text-to-image generation\n")

    def run_pipeline(
        self, 
        text_inputs, 
        evaluate=True, 
        save_outputs=True, 
        output_dir="pipeline_outputs"
    ):
        """
        Run complete end-to-end pipeline

        Args:
            text_inputs: List of text prompts or single prompt
            evaluate: Whether to evaluate quality
            save_outputs: Whether to save results
            output_dir: Directory to save outputs

        Returns:
            Dictionary with images and metrics
        """
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]

        print("\n" + "="*70)
        print("🚀 RUNNING GENERATIVE AI PIPELINE")
        print("="*70)
        print(f"Number of prompts: {len(text_inputs)}")
        print(f"Evaluation: {'Enabled' if evaluate else 'Disabled'}")
        print(f"Save outputs: {'Yes' if save_outputs else 'No'}")
        print("="*70 + "\n")

        # Step 1: Generate images
        print("STEP 1: Image Generation")
        print("-" * 70)
        generated = self.system.generate_batch(
            text_inputs, num_inference_steps=30
        )
        images = [img for _, img in generated]

        # Step 2: Evaluate quality
        metrics = {}
        if evaluate and len(images) > 0:
            print("\nSTEP 2: Quality Evaluation")
            print("-" * 70)
            metrics = self.system.evaluate_quality(images)

        # Step 3: Display results
        print("STEP 3: Displaying Results")
        print("-" * 70)
        self.system.display_images(generated, figsize=(5*len(images), 5))

        # Step 4: Save outputs
        if save_outputs:
            print("\nSTEP 4: Saving Outputs")
            print("-" * 70)
            self.system.save_generation_report(
                text_inputs, images, metrics, output_dir
            )

        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")

        return {
            'prompts': text_inputs,
            'images': images,
            'metrics': metrics
        }