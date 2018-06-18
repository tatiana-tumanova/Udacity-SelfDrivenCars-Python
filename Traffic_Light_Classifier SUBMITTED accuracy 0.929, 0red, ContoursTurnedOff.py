
# coding: utf-8

# # Traffic Light Classifier
# ---
# 
# In this project, you’ll use your knowledge of computer vision techniques to build a classifier for images of traffic lights! You'll be given a dataset of traffic light images in which one of three lights is illuminated: red, yellow, or green.
# 
# In this notebook, you'll pre-process these images, extract features that will help us distinguish the different types of images, and use those features to classify the traffic light images into three classes: red, yellow, or green. The tasks will be broken down into a few sections:
# 
# 1. **Loading and visualizing the data**. 
#       The first step in any classification task is to be familiar with your data; you'll need to load in the images of traffic lights and visualize them!
# 
# 2. **Pre-processing**. 
#     The input images and output labels need to be standardized. This way, you can analyze all the input images using the same classification pipeline, and you know what output to expect when you eventually classify a *new* image.
#     
# 3. **Feature extraction**. 
#     Next, you'll extract some features from each image that will help distinguish and eventually classify these images.
#    
# 4. **Classification and visualizing error**. 
#     Finally, you'll write one function that uses your features to classify *any* traffic light image. This function will take in an image and output a label. You'll also be given code to determine the accuracy of your classification model.    
#     
# 5. **Evaluate your model**.
#     To pass this project, your classifier must be >90% accurate and never classify any red lights as green; it's likely that you'll need to improve the accuracy of your classifier by changing existing features or adding new features. I'd also encourage you to try to get as close to 100% accuracy as possible!
#     
# Here are some sample images from the dataset (from left to right: red, green, and yellow traffic lights):
# <img src="images/all_lights.png" width="50%" height="50%">
# 

# ---
# ### *Here's what you need to know to complete the project:*
# 
# Some template code has already been provided for you, but you'll need to implement additional code steps to successfully complete this project. Any code that is required to pass this project is marked with **'(IMPLEMENTATION)'** in the header. There are also a couple of questions about your thoughts as you work through this project, which are marked with **'(QUESTION)'** in the header. Make sure to answer all questions and to check your work against the [project rubric](https://review.udacity.com/#!/rubrics/1213/view) to make sure you complete the necessary classification steps!
# 
# Your project submission will be evaluated based on the code implementations you provide, and on two main classification criteria.
# Your complete traffic light classifier should have:
# 1. **Greater than 90% accuracy**
# 2. ***Never* classify red lights as green**
# 

# # 1. Loading and Visualizing the Traffic Light Dataset
# 
# This traffic light dataset consists of 1484 number of color images in 3 categories - red, yellow, and green. As with most human-sourced data, the data is not evenly distributed among the types. There are:
# * 904 red traffic light images
# * 536 green traffic light images
# * 44 yellow traffic light images
# 
# *Note: All images come from this [MIT self-driving car course](https://selfdrivingcars.mit.edu/) and are licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).*

# ### Import resources
# 
# Before you get started on the project code, import the libraries and resources that you'll need.

# ## Training and Testing Data
# 
# All 1484 of the traffic light images are separated into training and testing datasets. 
# 
# * 80% of these images are training images, for you to use as you create a classifier.
# * 20% are test images, which will be used to test the accuracy of your classifier.
# * All images are pictures of 3-light traffic lights with one light illuminated.
# 
# ## Define the image directories
# 
# First, we set some variables to keep track of some where our images are stored:
# 
#     IMAGE_DIR_TRAINING: the directory where our training image data is stored
#     IMAGE_DIR_TEST: the directory where our test image data is stored

# ## Load the datasets
# 
# These first few lines of code will load the training traffic light images and store all of them in a variable, `IMAGE_LIST`. This list contains the images and their associated label ("red", "yellow", "green"). 
# 
# You are encouraged to take a look at the `load_dataset` function in the helpers.py file. This will give you a good idea about how lots of image files can be read in from a directory using the [glob library](https://pymotw.com/2/glob/). The `load_dataset` function takes in the name of an image directory and returns a list of images and their associated labels. 
# 
# For example, the first image-label pair in `IMAGE_LIST` can be accessed by index: 
# ``` IMAGE_LIST[0][:]```.
# 

# In[1]:


import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

get_ipython().run_line_magic('matplotlib', 'inline')

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)


# ## Visualize the Data
# 
# The first steps in analyzing any dataset are to 1. load the data and 2. look at the data. Seeing what it looks like will give you an idea of what to look for in the images, what kind of noise or inconsistencies you have to deal with, and so on. This will help you understand the image dataset, and **understanding a dataset is part of making predictions about the data**.

# ---
# ### Visualize the input images
# 
# Visualize and explore the image data! Write code to display an image in `IMAGE_LIST`:
# * Display the image
# * Print out the shape of the image 
# * Print out its corresponding label
# 
# See if you can display at least one of each type of traffic light image – red, green, and yellow — and look at their similarities and differences.

# In[ ]:


## TODO: Write code to display an image in IMAGE_LIST (try finding a yellow traffic light!)
## TODO: Print out 1. The shape of the image and 2. The image's label

# The first image in IMAGE_LIST is displayed below (without information about shape or label)
#selected_image = IMAGE_LIST[0][0]
#plt.imshow(selected_image)


# # 2. Pre-process the Data
# 
# After loading in each image, you have to standardize the input and output!
# 
# ### Input
# 
# This means that every input image should be in the same format, of the same size, and so on. We'll be creating features by performing the same analysis on every picture, and for a classification task like this, it's important that **similar images create similar features**! 
# 
# ### Output
# 
# We also need the output to be a label that is easy to read and easy to compare with other labels. It is good practice to convert categorical data like "red" and "green" to numerical data.
# 
# A very common classification output is a 1D list that is the length of the number of classes - three in the case of red, yellow, and green lights - with the values 0 or 1 indicating which class a certain image is. For example, since we have three classes (red, yellow, and green), we can make a list with the order: [red value, yellow value, green value]. In general, order does not matter, we choose the order [red value, yellow value, green value] in this case to reflect the position of each light in descending vertical order.
# 
# A red light should have the  label: [1, 0, 0]. Yellow should be: [0, 1, 0]. Green should be: [0, 0, 1]. These labels are called **one-hot encoded labels**.
# 
# *(Note: one-hot encoding will be especially important when you work with [machine learning algorithms](https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/)).*
# 
# <img src="images/processing_steps.png" width="80%" height="80%">
# 

# ---
# <a id='task2'></a>
# ### (IMPLEMENTATION): Standardize the input images
# 
# * Resize each image to the desired input size: 32x32px.
# * (Optional) You may choose to crop, shift, or rotate the images in this step as well.
# 
# It's very common to have square input sizes that can be rotated (and remain the same size), and analyzed in smaller, square patches. It's also important to make all your images the same size so that they can be sent through the same pipeline of classification steps!

# In[2]:


## HELPERS ##

def count_non_null(lst):
    """Helper that takes in a 2D list. Required for masked images, to exclude the empty pixels"""
    total_nonnull = 0
    total_amount = 0
    for row in lst:
        for i in row:
            if i!=0: 
                total_nonnull += 1
                total_amount += i
    return (total_nonnull, total_amount)


# In[42]:


def remove_bckg_for_threshold(rgb_image, threshold, debug=False, visualdebug=False):
    #TODO might need to crop so that the edges of the traffic light touch the edges of the image; otherwise the 'outer' contour includes the whole image
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)[1] 
    
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #cv2.CHAIN_APPROX_SIMPLE)
    
    ###cnts = contours[1] # num of element depends on cv2.__version__; each element is a point in a contour: (col,row)
    
    contoured = rgb_image.copy()
    
    shape = rgb_image.shape
    max_x = shape[1]-1
    max_y = shape[0]-1
    
    # Create empty 2D mask; mask is black, polygon is white
    mask = np.zeros([max_y, max_x])
    
    # loop over the contours, and accumulate polygons they enclose in 1 common mask
    # the output of cv2.findContours is an array, with these elements: 
    # [[an image] [array of contours][array of references to parent/child contours that they call hierarchies]
    for i in range(len(contours[1])):
        c = contours[1][i] # num of element depends on cv2.__version__; each element is a point in a contour: (col,row)
        
        # TODO *if* I find images with >1 contour, will need to ignore all contours except for the parent (position #4 in hie array)
        
        #if len(contours)==3: hie = contours[2][i]
        #else: hie = []
        #if debug: print("hie of contour #",i,": ",hie)
        
        #x_positions = c[:,:,0]
        #y_positions = c[:,:,1]

        #if hie[3] == -1:
        #if (0 in x_positions) or (0 in y_positions) or (max_x in x_positions) or (max_y in y_positions):
        if True:
            cv2.drawContours(contoured, [c], -1, (0, random.randint(50,255), 0), 1)
            
            c_mask = np.zeros([max_y, max_x])
            # adding this contour to a sub-mask
            cv2.fillConvexPoly(c_mask, c, True, 255)
            
            # overlap the mask of this contour with other contours' masks. TODO need to only keep the outer contour
            for i in range(len(mask)):
                for j in range(len(mask[0])):
                    mask[i][j] = mask[i][j] + c_mask[i][j]
        #else: cv2.drawContours(contoured, [c], -1, (0, 0, random.randint(50,255)), 1)
    
    masked_image = np.copy(rgb_image)
    
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j].all() == 0: masked_image[i][j] = [0, 0, 0]
       
    if debug:
        print("Number of contours found ",len(contours[1]))
    if visualdebug:
        f, (p1,p4,p5,p6,p7) = plt.subplots(1, 5, figsize=(200,100))
        p1.imshow(rgb_image)
        #p2.imshow(gray, cmap='gray')
        #p3.imshow(blurred, cmap='gray')
        p4.imshow(thresh, cmap='gray')
        p5.imshow(contoured)
        p6.imshow(mask)
        p7.imshow(masked_image)
    
    
    return masked_image
    
    
def remove_bckg_by_contours(rgb_image, debug=False):
    masked = remove_bckg_for_threshold(rgb_image, 130, debug, False)
    masked_gray = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
    (total_nonnull, total_amount) = count_non_null(masked_gray)
    ratio_masked = total_nonnull / ( masked.shape[0]*masked.shape[1] )
    if ratio_masked > 0.5: return masked
    
    if debug: print("remove_bckg_by_contours: the mask is too small, ratio_masked ",ratio_masked,"; trying another threshold ")
    masked = remove_bckg_for_threshold(rgb_image, 170, debug)
    masked_gray = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
    (total_nonnull, total_amount) = count_non_null(masked_gray)
    ratio_masked = total_nonnull / ( masked.shape[0]*masked.shape[1] )
    if ratio_masked > 0.5: return masked
    else: return rgb_image # removal of the background failed 


# In[46]:


### t e s t ###
def test_remove_bckg_for_threshold(im_index):
    remove_bckg_for_threshold(IMAGE_LIST[im_index][0], 130, True, True)
    
    
#i = random.randint(0, len(IMAGE_LIST)-1)
#i = 189
i = 159
tst_masked_im = test_remove_bckg_for_threshold( i )


# In[8]:


# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    ## TODO: Resize image and pre-process so that all "standard" images are the same size  
    
    # turning off background removal for the moment - it decreases the accuracy 
    #foreground = remove_bckg_by_contours(image)
    
    #cropping
    #numcol = image.shape[1]
    #col_crop = int(round(numcol/4))
    #cropped_im = image[:, col_crop:-col_crop, :]
    
    standard_im = cv2.resize(image, (32, 32))
    
    return standard_im





# In[9]:


### t e s t ###
def test_standardize_input(im_index):
    orig = IMAGE_LIST[im_index][0]
    standardized_im = standardize_input(orig)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(200,100))
    ax1.imshow(orig)
    ax2.imshow(standardized_im)
 
    
test_standardize_input(66)


# ## Standardize the output
# 
# With each loaded image, we also specify the expected output. For this, we use **one-hot encoding**.
# 
# * One-hot encode the labels. To do this, create an array of zeros representing each class of traffic light (red, yellow, green), and set the index of the expected class number to 1. 
# 
# Since we have three classes (red, yellow, and green), we have imposed an order of: [red value, yellow value, green value]. To one-hot encode, say, a yellow light, we would first initialize an array to [0, 0, 0] and change the middle value (the yellow value) to 1: [0, 1, 0].
# 

# ---
# <a id='task3'></a>
# ### (IMPLEMENTATION): Implement one-hot encoding

# In[10]:


## TODO: One hot encode an image label
## Given a label - "red", "green", or "yellow" - return a one-hot encoded label

# Examples: 
# one_hot_encode("red") should return: [1, 0, 0]
# one_hot_encode("yellow") should return: [0, 1, 0]
# one_hot_encode("green") should return: [0, 0, 1]

def one_hot_encode(new_label_string, new_label_prob=1, current_label=[0,0,0]):
    """Summirizes the current estimations of the color with the new one. The output is *not* normalized"""
    d = {"red" : [1,0,0], "yellow" : [0,1,0], "green" : [0,0,1]}
    ## TODO: Create a one-hot encoded label that works for all classes of traffic lights 
    
    new_label = d[new_label_string]
    new_label_weighted = [x*new_label_prob for x in new_label] 
    result_label = [sum(x) for x in zip(current_label, new_label_weighted)]
    
    return result_label



# In[6]:


### t e s t ###
print (one_hot_encode('green',0.9, [1,0,0]))



# ### Testing as you Code
# 
# After programming a function like this, it's a good idea to test it, and see if it produces the expected output. **In general, it's good practice to test code in small, functional pieces, after you write it**. This way, you can make sure that your code is correct as you continue to build a classifier, and you can identify any errors early on so that they don't compound.
# 
# All test code can be found in the file `test_functions.py`. You are encouraged to look through that code and add your own testing code if you find it useful!
# 
# One test function you'll find is: `test_one_hot(self, one_hot_function)` which takes in one argument, a one_hot_encode function, and tests its functionality. If your one_hot_label code does not work as expected, this test will print ot an error message that will tell you a bit about why your code failed. Once your code works, this should print out TEST PASSED.

# In[11]:


### t e s t ###
# Importing the tests
import test_functions
tests = test_functions.Tests()

# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)


# ## Construct a `STANDARDIZED_LIST` of input images and output labels.
# 
# This function takes in a list of image-label pairs and outputs a **standardized** list of resized images and one-hot encoded labels.
# 
# This uses the functions you defined above to standardize the input and output, so those functions must be complete for this standardization to work!
# 

# In[12]:


def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)


# ## Visualize the standardized data
# 
# Display a standardized image from STANDARDIZED_LIST and compare it with a non-standardized image from IMAGE_LIST. Note that their sizes and appearance are different!

# # 3. Feature Extraction
# 
# You'll be using what you now about color spaces, shape analysis, and feature construction to create features that help distinguish and classify the three types of traffic light images.
# 
# You'll be tasked with creating **one feature** at a minimum (with the option to create more). The required feature is **a brightness feature using HSV color space**:
# 
# 1. A brightness feature.
#     - Using HSV color space, create a feature that helps you identify the 3 different classes of traffic light.
#     - You'll be asked some questions about what methods you tried to locate this traffic light, so, as you progress through this notebook, always be thinking about your approach: what works and what doesn't?
# 
# 2. (Optional): Create more features! 
# 
# Any more features that you create are up to you and should improve the accuracy of your traffic light classification algorithm! One thing to note is that, to pass this project you must **never classify a red light as a green light** because this creates a serious safety risk for a self-driving car. To avoid this misclassification, you might consider adding another feature that specifically distinguishes between red and green lights.
# 
# These features will be combined near the end of his notebook to form a complete classification algorithm.

# ## Creating a brightness feature 
# 
# There are a number of ways to create a brightness feature that will help you characterize images of traffic lights, and it will be up to you to decide on the best procedure to complete this step. You should visualize and test your code as you go.
# 
# Pictured below is a sample pipeline for creating a brightness feature (from left to right: standardized image, HSV color-masked image, cropped image, brightness feature):
# 
# <img src="images/feature_ext_steps.png" width="70%" height="70%">
# 

# ## RGB to HSV conversion
# 
# Below, a test image is converted from RGB to HSV colorspace and each component is displayed in an image.

# In[ ]:


### t e s t ###
# Convert and image to HSV colorspace
# Visualize the individual color channels

image_num = 0
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# Print image label
print('Label [red, yellow, green]: ' + str(test_label))

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')


# ---
# <a id='task7'></a>
# ### (IMPLEMENTATION): Create a brightness feature that uses HSV color space
# 
# Write a function that takes in an RGB image and returns a 1D feature vector and/or single value that will help classify an image of a traffic light. The only requirement is that this function should apply an HSV colorspace transformation, the rest is up to you. 
# 
# From this feature, you should be able to estimate an image's label and classify it as either a red, green, or yellow traffic light. You may also define helper functions if they simplify your code.

# In[13]:


## HELPERS ##


def split_image_horizontally(rgb_image, debug=False):
    rows_total = rgb_image.shape[0]
    rows_slice = int(round(rows_total/3))
    
    top_img = rgb_image[:rows_slice,:,:]
    mid_img = rgb_image[rows_slice:2*rows_slice,:,:]
    bottom_img = rgb_image[2*rows_slice:,:,:]
    
    return (top_img, mid_img, bottom_img)

lower_red0 = np.array([0, 30, 40])
upper_red0 = np.array([25, 256, 256])
lower_red1 = np.array([130, 30, 40])
upper_red1 = np.array([180, 256, 256])

#lower_green = np.array([35, 40, 40])
lower_green = np.array([35, 52, 40])
upper_green = np.array([130, 256, 256])

#lower_yellow = np.array([25, 40, 40])
lower_yellow = np.array([11, 40, 40])
upper_yellow = np.array([35, 256, 256])
 
def mask_for_bright_pixels(rgb_image, color, debug=False):
    """Masks so taht only the brightest pixels, where the color is definite, are kept"""
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hue = hsv_image[:,:,0] 
    if color=='red': 
        mask0 = cv2.inRange(hsv_image, lower_red0, upper_red0)
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask = mask0 + mask1
    elif color=='green': mask = cv2.inRange(hsv_image, lower_green, upper_green)
    elif color=='yellow': mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    else: return Nune
    masked_image = np.copy(rgb_image)
    masked_image[mask == 0] = [0, 0, 0]
    
    hsv_masked = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
    (total_nonnull, total_amount) = count_non_null(hsv_masked[:,:,0])
    return (masked_image, total_nonnull, total_amount)

def get_probability_by_avg_intencity(avg_intencity, debug=False):
    """Can be used for brigtness or saturation"""
    avg_threshold1 = 99
    avg_threshold2 = 119
    
    if avg_intencity >= avg_threshold2: probability = 1
    elif avg_intencity >= avg_threshold1: probability = 0.5
    else: probability = 0.3
        
    if debug: print("get_probability_by_avg_intencity avg_intencity ",avg_intencity,"; avg_threshold1 ",avg_threshold1,", avg_threshold2 ",avg_threshold2,"; probability ",probability)
        
    return probability

def mask_for_avg_color(rgb_image, debug=False):
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    max_sat = np.amax(hsv_image[:,:,1])
    #if debug: print("mask_n_crop max_sat = ",max_sat)
    if max_sat > 220: upper_sat = 180
    elif max_sat > 150: upper_sat = 120
    elif max_sat > 50: upper_sat = 30
    else: upper_sat = 10
    
    lower_saturation = np.array([0, 0, 0])
    upper_saturation = np.array([180, upper_sat, 256]) 
    mask = cv2.inRange(hsv_image, lower_saturation, upper_saturation)
    masked_image = np.copy(rgb_image)
    masked_image[mask != 0] = [0, 0, 0]
    
    #edge case: whole image masked (saturation too low)
    if np.sum(masked_image) == 0:
        if debug: print("mask_n_crop failure, saturation too low")
        masked_image = rgb_image
    

    return masked_image


# In[27]:


### t e s t ###
def test_split_image_horizontally(im_index):
    im = standardize_input(IMAGE_LIST[im_index][0])
    
    (top_img, mid_img, bottom_img)= split_image_horizontally(im)
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
    ax1.imshow(im)
    ax2.imshow(top_img)
    ax3.imshow(mid_img)
    ax4.imshow(bottom_img)
    
test_split_image_horizontally(66)


# In[ ]:



    


# In[14]:


## 1 Classify by brightest pixels color in each of 3 image slices, if clear color is at all present


def create_brightly_colored_pixels_feature(rgb_image, current_label=[0,0,0], debug=False):
    """ This algorithm produces false-positive replies with images that have bright green/yellow and pale red. 
    So it's the price for never missing a red light """
    
    masked_for_avgbr_im = mask_for_avg_color(rgb_image, debug)
    hsv_image = cv2.cvtColor(masked_for_avgbr_im, cv2.COLOR_RGB2HSV)
    hue = hsv_image[:,:,0]
        
    (total_nonnull, total_hue) = count_non_null(hue)    
    avg_hue = total_hue/total_nonnull
    probability = get_probability_by_avg_intencity(avg_hue, debug)
     
    
    
    (top_img, mid_img, bottom_img)= split_image_horizontally(rgb_image)
    #######TODO could take into account both number and average saturation/brightness. 
    ###so the mask_for_bright_pixels must return masked rgb.
    #####also, teh pixels closer to teh center could be given more value
    masked_red, nn_red, total_red  = mask_for_bright_pixels(top_img, 'red', debug)
    masked_yellow, nn_yellow, total_yellow  = mask_for_bright_pixels(mid_img, 'yellow', debug)
    masked_green, nn_green, total_green  = mask_for_bright_pixels(bottom_img, 'green', debug)
    
    color = 'undefined'
    if (nn_red>5 and nn_green<50 and nn_yellow<50) or (nn_red>15 and nn_green<100 and nn_yellow<100) or (nn_red>100): 
        color = 'red'        
    elif (nn_red<25 and nn_yellow>=60) or (nn_red<5 and nn_yellow>=30) or (nn_red==0 and nn_yellow>=5):
        color = 'yellow'
    elif (nn_red<25 and nn_green>=60) or (nn_red<5 and nn_green>=30) or (nn_red==0 and nn_green>=5):
        color = 'green'            
    
        
    if debug: print("create_brightly_colored_pixels_feature: detected ",color,", probability ",probability,"; (nonnull/total r ",nn_red,"/",total_red,", nonnull/total g ",nn_green,"/",total_green,", nonnull/total y ",nn_yellow,"/",total_yellow,")")
        
    if color != 'undefined': return one_hot_encode(color, probability, current_label)
    else: return [0,0,0]


# In[15]:


### t e s t ###
def tst_brightly_colored_pixels(img_number, dataset):
    if dataset=='test':
        raw_images = helpers.load_dataset(IMAGE_DIR_TEST)        
    elif dataset=='training':
        raw_images = helpers.load_dataset(IMAGE_DIR_TRAINING)
    images = standardize(raw_images)
    
    im = images[img_number][0]
    (top_img, mid_img, bottom_img)= split_image_horizontally(im)    
    
    (masked_red, total_nonnull_red, total_amount_red) = mask_for_bright_pixels(top_img, 'red', True)
    (masked_yellow, total_nonnull_yellow, total_amount_yellow) = mask_for_bright_pixels(mid_img, 'yellow', True)
    (masked_green, total_nonnull_green, total_amount_green) = mask_for_bright_pixels(bottom_img, 'green', True)
    
    color = create_brightly_colored_pixels_feature(im, [0,0,0],True)
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
    ax1.imshow(im)
    ax2.imshow(masked_red)
    ax3.imshow(masked_yellow)
    ax4.imshow(masked_green)
    print("Detected by brigth pixels: ",color)
    
tst_brightly_colored_pixels(6, 'test')



# In[16]:


## 2 Classify by area of bright pixels concentration

## TODO: Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
## This feature should use HSV colorspace values

    
## TODO: Convert image to HSV color space
## TODO: Create and return a feature value and/or vector


def create_position_feature(rgb_image, channel, current_label=[0,0,0], max_possible_probability=1, debug=False):
    """Channel: 1 for saturation, 2 for brightness"""
    (top_img, mid_img, bottom_img)= split_image_horizontally(rgb_image)
    
    top_img_hsv = cv2.cvtColor(top_img, cv2.COLOR_RGB2HSV)
    mid_img_hsv = cv2.cvtColor(mid_img, cv2.COLOR_RGB2HSV)
    bottom_img_hsv = cv2.cvtColor(bottom_img, cv2.COLOR_RGB2HSV)
    
    
    if channel=='both': total_top = np.sum( top_img_hsv[:,:,1] ) + np.sum( top_img_hsv[:,:,2] )
    else: total_top = np.sum( top_img_hsv[:,:,channel] )
    
    if channel=='both': total_mid = np.sum( mid_img_hsv[:,:,1] ) + np.sum( mid_img_hsv[:,:,2] )
    else: total_mid = np.sum( mid_img_hsv[:,:,channel] )
        
    if channel=='both': total_bottom = np.sum( bottom_img_hsv[:,:,1] ) + np.sum( bottom_img_hsv[:,:,2] )
    else: total_bottom = np.sum( bottom_img_hsv[:,:,channel] )


    t2m = total_top/total_mid
    t2b = total_top/total_bottom
    
    m2t = total_mid/total_top
    m2b = total_mid/total_bottom
    
    b2t = total_bottom/total_top
    b2m = total_bottom/total_mid
    
    #threshold = 1.046#1.03 this worked for H or S
    threshold1 = 1.046
    prob1 = 1
    threshold2 = 1.02
    prob2 = 0.4
    threshold3 = 1.015
    prob3 = 0.1
    
    color = 'undefined'
    
    if t2m>threshold1 and t2b>threshold1:
        color = 'red'
        probability = 1
    elif t2m>threshold2 and t2b>threshold2:
        color = 'red'
        probability = prob2
    elif t2m>threshold3 and t2b>threshold3:
        color = 'red'
        probability = 0.1
        
    elif m2t>threshold1 and m2b>threshold1:
        color = 'yellow'
        probability = 1
    elif m2t>threshold2 and m2b>threshold2:
        color = 'yellow'
        probability = prob2
    elif m2t>threshold3 and m2b>threshold3:
        color = 'yellow'
        probability = 0.1
        
    elif b2t>threshold1 and b2m>threshold1:
        color = 'green'
        probability = 1
    elif b2t>threshold2 and b2m>threshold2:
        color = 'green'
        probability = prob2
    elif b2t>threshold3 and b2m>threshold3:
        color = 'green'
        probability = 0.1
    
    else: probability = 0
        
    probability = probability * max_possible_probability
        
    if debug: 
        print("\ncreate_position_feature: channel *",channel,"*; top ", total_top,"; mid ",total_mid,"; bottom ",total_bottom)
        print("t2m=",t2m,", t2b=",t2b, "; m2t=",m2t,", m2b=",m2b, "; b2t=",b2t,", b2m=",b2m)
        print("create_position_feature: color detected ",color,", probability=",probability,"\n")
    
    if color != 'undefined': return one_hot_encode(color, probability, current_label)
    else: return current_label

def create_saturation_position_feature(rgb_image, current_label=[0,0,0], debug=False):
    return create_position_feature(rgb_image, 1, current_label, 1, debug)

def create_brightness_position_feature(rgb_image, current_label=[0,0,0], debug=False):
    return create_position_feature(rgb_image, 2, current_label, 1, debug)


def create_SV_position_feature(rgb_image, current_label=[0,0,0], debug=False):
    return create_position_feature(rgb_image, 'both', current_label, 1, debug)


# In[17]:


### t e s t ###

def tst_create_SV_position_feature(img_number):
    im = STANDARDIZED_LIST[img_number][0]
    create_SV_position_feature(im, [0,0,0], True)

    plt.imshow(im)
    

    
tst_create_SV_position_feature(779)


# In[18]:


## 3 Classify by average color (only the pixels with reasonably high saturation level are taken into account)
def create_avg_color_feature(rgb_image, current_label=[0,0,0], debug=False):
    cropped_image = mask_for_avg_color(rgb_image, debug)
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)
    hue = hsv_image[:,:,0]
    (total_nonnull, total_hue) = count_non_null(hue)    
    avg_hue = total_hue/total_nonnull
    
    #we wont to keep the same intencity thresholds, but decrease the probability since avg_color is the least reliable classifier 
    probability = get_probability_by_avg_intencity(avg_hue)/1.5 
    
    if debug: print("create_color_feature: total_nonnull=",total_nonnull,"; total_hue=",total_hue,"; avg_hue=",avg_hue)
    
    if 0<avg_hue<=80: color = 'yellow'   
    elif 80<avg_hue<=130: color = 'green'
    else: color = 'red'
                            
    if debug: print("create_color_feature avg_hue=",avg_hue,"; detecting ",color)
    return one_hot_encode(color, probability, current_label)

    






# In[19]:


### t e s t ###
    
def tst_create_avg_color_feature(img_number):
    im = STANDARDIZED_LIST[img_number][0]
    masked = mask_for_avg_color(im)
    color = create_avg_color_feature(im, [0,0,0], True)
    print("Img_number=",img_number,"; detected ",color,"; actual ", STANDARDIZED_LIST[img_number][1])
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(im)
    ax2.imshow(masked)
 

im_number = 749

tst_create_avg_color_feature(im_number)
#tst_mask_for_bright_pixels(im_number)


# ## (Optional) Create more features to help accurately label the traffic light images

# In[ ]:


# (Optional) Add more image analysis and create more features


# ## (QUESTION 1): How do the features you made help you distinguish between the 3 classes of traffic light images?

# **Answer:**
# There are 3 features:
# 
# 1) Determine the color by total amount of brightly colored pixels. Each of the 3 horizontal slices is estimated separately. The results for each slice are compared, and 'red' is given higher weight since we cannot effort taking red for anything else
# 
# 2) Determine by combined intencity of S, V channels. Each of the 3 horizontal slices is estimated separately, and the decision is made by the position of the highest contentration (top/mid/bottom). Often S and V produce the opposite results, so the combination of the two is zero. I've tried estimating the S and V separately, and combining the results with various weights, but it gives less precision
# 
# 3) Determine by average brightness of selected pixels. This is the only classifier which estimates the image as a whole, not trying to slice it into top/mid/bottom slices
# 
# Each classifier adjusts the probability of the defected color taking into account secondary features such as average brightness of the masked area.
# The output of each classifier is combined, and then is normalized.
# 
# With the current dataset, the results are
# Accuracy: 0.9292929292929293
# Number of misclassified images = 21 out of 297
# ..and 0 occurances of 'red' detected as 'green' in training or test.

# # 4. Classification and Visualizing Error
# 
# Using all of your features, write a function that takes in an RGB image and, using your extracted features, outputs whether a light is red, green or yellow as a one-hot encoded label. This classification function should be able to classify any image of a traffic light!
# 
# You are encouraged to write any helper functions or visualization code that you may need, but for testing the accuracy, make sure that this `estimate_label` function returns a one-hot encoded label.

# ---
# <a id='task8'></a>
# ### (IMPLEMENTATION): Build a complete classifier 

# In[20]:


# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label

## TODO: Extract feature(s) from the RGB image and use those features to
## classify the image and output a one-hot encoded label


def estimate_label(rgb_image, debug=False):
    # determine by pixels that have clear color
    predicted_label = create_brightly_colored_pixels_feature(rgb_image)
    if debug: print("\n*\nestimate_label bright pixels ",predicted_label)
    
    # determine by position (by horizontal slice of the image wiht high concentration of colored pixels): saturation
    #predicted_label = create_saturation_position_feature(rgb_image, predicted_label)   
    # determine by position (by horizontal slice of the image wiht high concentration of colored pixels): brightness
    #predicted_label = create_brightness_position_feature(rgb_image, predicted_label)
    #if predicted_label != None: return predicted_label   
    
    predicted_label = create_SV_position_feature(rgb_image, predicted_label)
    if debug: print("estimate_label bright pixels + SV position ",predicted_label) 
    
    # last attempt: guess-estimate by average color or a higher saturated area
    predicted_label = create_avg_color_feature(rgb_image, predicted_label)    
    if debug: print("estimate_label bright pixels + SV position + avg color ",predicted_label)
        
    
    #normalization
    r = predicted_label[0]
    y = predicted_label[1]
    g = predicted_label[2]
    
    if r>=y and r>=g: return one_hot_encode('red')
    elif y>r and y>g: return one_hot_encode('yellow')
    else: return one_hot_encode('green')
    
   
    
    


# In[21]:


### t e s t ###
def tst_estimate_label(im_num, dataset='test'):
    if dataset=='test':
        raw_images = helpers.load_dataset(IMAGE_DIR_TEST)        
    elif dataset=='training':
        raw_images = helpers.load_dataset(IMAGE_DIR_TRAINING)
    images = standardize(raw_images)
    

    image = images[im_num]
    im = image[0]
    true_label = image[1]
    predicted_label = estimate_label(im)
    
    print(predicted_label)
    
tst_estimate_label(0)


# ## Testing the classifier
# 
# Here is where we test your classification algorithm using our test set of data that we set aside at the beginning of the notebook! This project will be complete once you've pogrammed a "good" classifier.
# 
# A "good" classifier in this case should meet the following criteria (and once it does, feel free to submit your project):
# 1. Get above 90% classification accuracy.
# 2. Never classify a red light as a green light. 
# 
# ### Test dataset
# 
# Below, we load in the test dataset, standardize it using the `standardize` function you defined above, and then **shuffle** it; this ensures that order will not play a role in testing accuracy.
# 

# In[22]:


# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)


# ## Determine the Accuracy
# 
# Compare the output of your classification algorithm (a.k.a. your "model") with the true labels and determine the accuracy.
# 
# This code stores all the misclassified images, their predicted labels, and their true labels, in a list called `MISCLASSIFIED`. This code is used for testing and *should not be changed*.

# In[31]:


def get_undetected_red(dataset='test'):
    if dataset=='test':
        raw_images = helpers.load_dataset(IMAGE_DIR_TEST)        
    elif dataset=='training':
        raw_images = helpers.load_dataset(IMAGE_DIR_TRAINING)
    images = standardize(raw_images)
    und_red = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for i in range(len(images)):
        image = images[i]
        # Get true data
        im = image[0]
        true_label = image[1]
        predicted_label = estimate_label(im)
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            if predicted_label==[0,0,1] and true_label==[1,0,0]: 
                und_red.append((im, predicted_label, true_label, i))
    
    
    # Accuracy calculations
    total = len(images)
    
    print("Number of undetected red in ",dataset," = ",len(und_red),' out of ',len(images))
    for i in und_red:
        print("\tImage num ",i[3]," in dataset ",dataset," false detected ",i[1])
    
    return und_red

undetected_red = get_undetected_red('training')
undetected_red = get_undetected_red('test')


# In[33]:


# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)


def get_misclassified_images(test_images):
    #undetected_red_training = get_undetected_red('training')
    #undetected_red.extend( get_undetected_red('test') )
    
    
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for i in range(len(test_images)):
        image = test_images[i]
        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
    
    
    
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total



print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))



# ---
# <a id='task9'></a>
# ### Visualize the misclassified images
# 
# Visualize some of the images you classified wrong (in the `MISCLASSIFIED` list) and note any qualities that make them difficult to classify. This will help you identify any weaknesses in your classification algorithm.

# In[26]:


# Visualize misclassified example(s)
## TODO: Display an image in the `MISCLASSIFIED` list 
## TODO: Print out its predicted label - to see what the image *was* incorrectly classified as
def visualize_misclassified(img_num, collection = 'missclassified'):
    if collection == 'undetected_red': coll = undetected_red
    else: coll = MISCLASSIFIED
        
    img = coll[img_num][0]
    true_label=coll[img_num][2]
    
    (top_img, mid_img, bottom_img)= split_image_horizontally(img)
    
    masked_red, nn_red, total_amount_red  = mask_for_bright_pixels(top_img, 'red', True)
    masked_yellow, nn_yellow, total_amount_yellow  = mask_for_bright_pixels(mid_img, 'yellow', True)
    masked_green, nn_green, total_amount_green  = mask_for_bright_pixels(bottom_img, 'green', True)
    
    color_by_bright_pixels = create_brightly_colored_pixels_feature(img, [0,0,0], True)

    color_by_saturation_position = create_saturation_position_feature(img, [0,0,0], True)
    color_by_brightness_position = create_brightness_position_feature(img, [0,0,0], True)
    color_by_SV_position = create_SV_position_feature(img, [0,0,0], True)
    
    masked_for_avg_color = mask_for_avg_color(img, True)
    color_by_average_color = create_avg_color_feature(img, [0,0,0], True)
    
    detected_label = estimate_label(img, True)

    print("\n***\nTrue label ",true_label)
    
    print("\nDetected color_by_bright_pixels ",color_by_bright_pixels)
    print("Detected color_by_saturation_position ",color_by_saturation_position)
    print("Detected color_by_brightness_position ",color_by_brightness_position)
    print("Detected color_by_SV_position ",color_by_SV_position)    
    print("Detected color_by_average_color ",color_by_average_color)
    
    print("\nDetected label ", detected_label)

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # HSV channels
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    f, (ax0, ax1r, ax1y, ax1g, avg, ax2, ax3, ax4) = plt.subplots(1, 8, figsize=(20,10))
    ax0.set_title('Standardized image')
    ax0.imshow(img)

    ax1r.set_title(nn_red)
    ax1r.imshow(masked_red)
    ax1y.set_title(nn_yellow)
    ax1y.imshow(masked_yellow)  
    ax1g.set_title(nn_green)
    ax1g.imshow(masked_green)   
    
    avg.set_title('masked_for_avg_color')
    avg.imshow(masked_for_avg_color)
    
    ax2.set_title('H channel')
    ax2.imshow(h, cmap='gray')
    ax3.set_title('S channel')
    ax3.imshow(s, cmap='gray')
    ax4.set_title('V channel')
    ax4.imshow(v, cmap='gray')
    
#visualize_misclassified(0, 'undetected_red')
visualize_misclassified(1)


# ---
# <a id='question2'></a>
# ## (Question 2): After visualizing these misclassifications, what weaknesses do you think your classification algorithm has? Please note at least two.

# **Answer:** Write your answer in this cell.
# 1. The bright or intensively saturated background really ruins classifier #2 (the one that classifies by combined intencity of S, V channels). I'd have better results if I could crop by the traffic light boundaries (find a darker colored square, or not square, and chop off the background). I've made several attempts to find the background by contours and remove it; it works relatively relyable for many images, but not for all of them. So had to turn it off.
# 2. The calculation highly relies on splitting the image into 3 horizontal slices. When the image is shifted vertically, it's much less precise. Again, cropping by the boundaries of the traffic light would help.
# 3. In an attempt to never classify 'red' as 'green', I had to tune the thresholds so that they do a lot of the opposite mistakes (in any uncertanty think 'red'). If we could have a series of images, we could add an extra criteria ("the previous light was green, so now it can be green or yellow, but not red).
# 4. Detection of yellow is not satisfactory; but changing the mask leads to mis-detecting yellowish background for a traffic light.
# 5. On a very sunny day, all 3 colors a visible and are relatively bright. By detecting such case, could eliminate 2-3 mistakes out of 2k (~0.1% improvement, not worth it). 

# ## Test if you classify any red lights as green
# 
# **To pass this project, you must not classify any red lights as green!** Classifying red lights as green would cause a car to drive through a red traffic light, so this red-as-green error is very dangerous in the real world. 
# 
# The code below lets you test to see if you've misclassified any red lights as green in the test set. **This test assumes that `MISCLASSIFIED` is a list of tuples with the order: [misclassified_image, predicted_label, true_label].**
# 
# Note: this is not an all encompassing test, but its a good indicator that, if you pass, you are on the right track! This iterates through your list of misclassified examples and checks to see if any red traffic lights have been mistakenly labelled [0, 1, 0] (green).

# In[34]:


# Importing the tests
import test_functions
tests = test_functions.Tests()

if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")


# # 5. Improve your algorithm!
# 
# **Submit your project after you have completed all implementations, answered all questions, AND when you've met the two criteria:**
# 1. Greater than 90% accuracy classification
# 2. No red lights classified as green
# 
# If you did not meet these requirements (which is common on the first attempt!), revisit your algorithm and tweak it to improve light recognition -- this could mean changing the brightness feature, performing some background subtraction, or adding another feature!
# 
# ---

# ### Going Further (Optional Challenges)
# 
# If you found this challenge easy, I suggest you go above and beyond! Here are a couple **optional** (meaning you do not need to implement these to submit and pass the project) suggestions:
# * (Optional) Aim for >95% classification accuracy.
# * (Optional) Some lights are in the shape of arrows; further classify the lights as round or arrow-shaped.
# * (Optional) Add another feature and aim for as close to 100% accuracy as you can get!
