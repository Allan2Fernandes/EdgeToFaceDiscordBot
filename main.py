import discord
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2


client = discord.Client(intents=discord.Intents.all())
model_path = "C:/Users/allan/PycharmProjects/Edge2Face/Models/Epoch5/Generator"
generator_model = tf.keras.models.load_model(model_path)


def get_img_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def plot_image(img):
    plt.imshow(img)
    plt.show()
    pass



def post_process_tensor(tensor):
    tensor = np.array(tensor)
    tensor = tensor*0.5+0.5

    #Convert to UInt8
    tensor = tensor*255
    tensor = tensor.astype(np.uint8)
    return tensor


def center_crop(img):
    #Get the 2 dimensions
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    #Get the smaller dim
    smaller_dim = min(dim1, dim2)

    #Get the center for both dimensions
    dim1_center = dim1 // 2
    dim2_center = dim2 // 2

    #Crop from center for both dimensions
    dim1_start = dim1_center-smaller_dim//2
    dim1_end = dim1_center+smaller_dim//2

    dim2_start = dim2_center-smaller_dim//2
    dim2_end = dim2_center+smaller_dim//2

    img = img[dim1_start:dim1_end, dim2_start:dim2_end]
    return img

def preprocess_tensor(input_tensor):
    processed_tensor = tf.convert_to_tensor(input_tensor)
    processed_tensor = tf.cast(processed_tensor, tf.float32)
    processed_tensor = (processed_tensor - 127.5) / 127.5
    return processed_tensor

def preprocess_input_image(image):
    #Conver the image to usable format
    image_tensor = np.array(image)
    #Center crop the image
    image_tensor = center_crop(img=image_tensor)
    #Resize the image to the correct dimensions for the generator
    image_tensor = cv2.resize(image_tensor, (256, 256), interpolation=cv2.INTER_CUBIC)
    #Blur the image to clean background edges
    blurred_image = cv2.GaussianBlur(image_tensor, (3, 3), 0)
    #Apply the edge detection filter
    edge_image = cv2.Canny(blurred_image, threshold1=140, threshold2=140)
    #Expand dims to input into generator
    edge_image = tf.expand_dims(tf.expand_dims(edge_image, axis = 0), axis = -1)
    #Expand channels to input into generator
    edge_image = tf.tile(edge_image, [1,1,1,3])
    return edge_image

@client.event
async def on_ready():
    print("We have logged in as {0.user}".format(client))
    pass

@client.event
async def on_message(message):
    if message.author == client.user: #You don't want it to reply to itself
        return

    if message.content.startswith('T'):
        #This is colour image
        img = get_img_from_url(message.attachments[0])
        #Translate to edges only
        img = preprocess_input_image(img)
        #Preprocess the tensor scale
        img = preprocess_tensor(img)
        #Pass the edges into generator
        reconstructed_image = generator_model(img)
        reconstructed_image = reconstructed_image[0]
        reconstructed_image = post_process_tensor(reconstructed_image)
        #Save the generated image locally
        image_file = Image.fromarray(reconstructed_image)
        image_file.save("Images/reconstructed_image.jpeg")
        #Attach the file in a message and send it in the channel
        await message.channel.send(file=discord.File("Images/reconstructed_image.jpeg"))
        pass
    pass



client.run('MTAwMjYxODA1Nzg5NTcxMDg3MQ.GCy2fJ.o13CAJMPf4hSLG-rS7yOqoUVwRa-7y01Rme--Y')
