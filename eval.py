import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)
import model
import cv2
from tqdm import tqdm

# NUM_IMAGES = 45406
NUM_IMAGES = 63825
OUTPUT_FILE = "output/predicted_angles.txt"

# Make sure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

# Get the true angle from the data.txt file
# Example line of data.txt: "23.jpg 11.700000"
true_angles_degrees = []
with open("driving_dataset/data.txt") as f:
    for line in f:
        true_angles_degrees.append(float(line.split(" ")[1]))
print(len(true_angles_degrees))
assert len(true_angles_degrees) == NUM_IMAGES, f"Mismatch between number of images {NUM_IMAGES} and angles in data.txt {len(true_angles_degrees)}"

# Open the output file once before the loop
with open(OUTPUT_FILE, "w") as f:
    # Create progress bar
    progress_bar = tqdm(range(NUM_IMAGES), desc="Processing images", unit="img")
    f.write(f"image_number,predicted_angle,true_angle\n")  # Write header to the file
    for i in progress_bar:
        full_image = cv2.imread("driving_dataset/" + str(i) + ".jpg")
        image = cv2.resize(full_image[-150:], (200, 66)) / 255.0
        predicted_angle_degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / 3.14159265

        
        # Update progress bar description every 100 images
        if i % 100 == 0:
            progress_bar.set_postfix({"angle": f"{predicted_angle_degrees:.2f}°", "img": i})
        
        # Write to the already-opened file
        f.write(f"{i},{predicted_angle_degrees},{true_angles_degrees[i]}\n")