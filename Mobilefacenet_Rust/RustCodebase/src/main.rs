use tract_onnx::prelude::*;
use image::{imageops::FilterType, GenericImageView, Pixel};
use std::env;
use anyhow::{Result, Context, anyhow};

fn main() -> Result<()> {
    // 1. Get command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_image>", args[0]);
        // Return error code so calling script knows it failed
        std::process::exit(1);
    }
    let image_path = &args[1];

    // 2. Load the model (Look in current directory or parent)
    let model_path = if std::path::Path::new("mobilefacenet.onnx").exists() {
        "mobilefacenet.onnx"
    } else {
        "../ConvertionToONNXx86/mobilefacenet.onnx"
    };

    // Load, optimize, and prepare for execution
    let model = tract_onnx::onnx()
        .model_for_path(model_path)
        .context("Failed to load ONNX model")?
        .into_optimized()?
        .into_runnable()?;

    // 3. Load the image from disk
    let img = image::open(image_path)
        .context(format!("Failed to open image at: {}", image_path))?;

    // 4. Preprocess (Resize to 112x112 -> Normalize)
    let tensor = preprocess_image(&img)?;

    // 5. Run Inference
    let result = model.run(tvec!(tensor.into()))?;

    // 6. Extract and L2 Normalize
    let embedding_tensor = &result[0];
    let embedding: Vec<f32> = embedding_tensor
        .to_array_view::<f32>()?
        .iter()
        .cloned()
        .collect();

    let normalized_embedding = l2_normalize(&embedding);

    // 7. Print ONLY the vector as JSON for easy Python parsing
    println!("{:?}", normalized_embedding);

    Ok(())
}

fn preprocess_image(img: &image::DynamicImage) -> Result<Tensor> {
    // Resize to 112x112 using Triangle filter (good balance of speed/quality)
    let resized = img.resize_exact(112, 112, FilterType::Triangle);
    
    // Convert to Tensor [1, 3, 112, 112] and normalize to [-1, 1]
    let tensor: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 112, 112), |(_, c, y, x)| {
        let pixel = resized.get_pixel(x as u32, y as u32);
        let val = pixel.channels()[c] as f32;
        (val - 127.5) / 128.0
    })
    .into();

    Ok(tensor)
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 { return v.to_vec(); }
    v.iter().map(|x| x / norm).collect()
}