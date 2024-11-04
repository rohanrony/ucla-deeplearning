from typing import Optional
import zipfile
import PIL.Image

import numpy as np
import tensorflow as tf

import pandas as pd

def load_images(segment: str, limit: Optional[int] = None):
    with zipfile.ZipFile('pnp_dataset.zip') as z:

        targets = sorted(p for p in z.namelist() if p.startswith(f'pnp_dataset/{segment}') and 'npy' not in p)
        if limit is not None:
            targets = targets[:limit]

        for _, target in enumerate(targets):
            with z.open(target) as f:
                image_pixels = np.array(PIL.Image.open(f), dtype=np.float32)
                yield image_pixels.astype(np.float16)

def load_labels(segment: str, limit: Optional[int] = None):
    with zipfile.ZipFile('pnp_dataset.zip') as z:
        with z.open(f'pnp_dataset/{segment}_y.npy') as f:
            train_y = np.load(f)
            if limit is not None:
                train_y = train_y[:limit]
            for label in train_y:
                yield label
              
def build_dataset(segment: str, limit: Optional[int] = None, include_labels: bool = True) -> tf.data.Dataset:
    model_input = tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)  # type: ignore
    model_output = tf.TensorSpec(shape=(), dtype=tf.int32)  # type: ignore

    def load():
        if include_labels:
            yield from zip(load_images(segment, limit), load_labels(segment, limit))
        else:
            yield from load_images(segment, limit)

    if include_labels:
        dataset_signature = (model_input, model_output)
    else:
        dataset_signature = model_input

    dataset = tf.data.Dataset.from_generator(load, output_signature=dataset_signature)
    return dataset


# See Python Generator
# https://peps.python.org/pep-0255/
def build_generator_labeled_paired(metadata: pd.DataFrame):
    def generator():
        for _, row in metadata.iterrows():
            training_path = 'fakenet_dataset/train/images/' + row['file_name_training']
            generated_path = 'fakenet_dataset/train/images/' + row['file_name_generated']

            training_np = np.array(Image.open(training_path))
            generated_np = np.array(Image.open(generated_path))

            model_output = np.random.randint(low=0, high=2, size=1)

            if model_output == 1:
                model_input = (training_np, generated_np)
            else:
                model_input = (generated_np, training_np)

            yield (model_input, model_output)

    return generator


# See Tensorflow Dataset
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset
def build_dataset_labeled_paired(metadata: pd.DataFrame):
    image_signature = tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)  # type: ignore

    model_input = (image_signature, image_signature)
    model_output = tf.TensorSpec(shape=(1,), dtype=tf.int32)  # type: ignore

    dataset_signature = (model_input, model_output)

    dataset = tf.data.Dataset.from_generator(
        build_generator_labeled_paired(metadata), 
        output_signature=dataset_signature
    )

    return dataset





def load_image(file_path: str) -> tf.Tensor:
    # Read and preprocess image
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def generate_samples(metadata: pd.DataFrame, image_dir: str):
    for _, row in metadata.iterrows():
        image_path = os.path.join(image_dir, row['file_name'])
        image = load_image(image_path)
        vehicle_count = tf.cast(row['vehicle'], tf.int32)
        signal_count = tf.cast(row['signal'], tf.int32)
        yield image, (vehicle_count, signal_count)

def build_dataset(metadata: pd.DataFrame, image_dir: str) -> tf.data.Dataset:
    # Define input and output specifications
    input_spec = tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)
    output_spec = (
        tf.TensorSpec(shape=(), dtype=tf.int32),  # vehicle count
        tf.TensorSpec(shape=(), dtype=tf.int32)   # signal count
    )
    
    # Create dataset signature
    dataset_signature = (input_spec, output_spec)
    
    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        lambda: generate_samples(metadata, image_dir),
        output_signature=dataset_signature
    )
    
    return dataset
