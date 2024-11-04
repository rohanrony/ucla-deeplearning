from typing import Optional
import zipfile
import PIL.Image
import os

import numpy as np
import tensorflow as tf

import pandas as pd


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
        
def generate_score_samples(metadata: pd.DataFrame, image_dir: str):
    for _, row in metadata.iterrows():
        image_path = os.path.join(image_dir, row['file_name'])
        image = load_image(image_path)
        yield image

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

def build_score_dataset(metadata: pd.DataFrame, image_dir: str) -> tf.data.Dataset:
    # Define input and output specifications
    input_spec = tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)
    
    
    # Create dataset signature
    dataset_signature = (input_spec)
    
    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        lambda: generate_score_samples(metadata, image_dir),
        output_signature=dataset_signature
    )
    
    return dataset