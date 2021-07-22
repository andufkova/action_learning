from flask import Flask, render_template, request, redirect, url_for, abort, session
from flask_dropzone import Dropzone
from werkzeug.utils import secure_filename
import random
import string
from pathlib import Path
import numpy as np
from tifffile import TiffFile
from joblib import load
import os
import glob
from dataset import LandCoverData as LCD
from infer import *
from model import UNet
import xgboost
import skvideo.io
import cv2
import tensorflow as tf
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

model_xgb = load('models/xgbmodel_trained_on_400.joblib')

#unet_kwargs = dict(
#        input_shape=(LCD.IMG_SIZE, LCD.IMG_SIZE, LCD.N_CHANNELS),
#        num_classes=LCD.N_CLASSES,
#        num_layers=2
#    )
#print(f"Creating U-Net with arguments: {unet_kwargs}")
#model_unet = UNet(**unet_kwargs)
#model_unet.load_weights('models/epoch12')

model_unet = tf.keras.models.load_model('models/epoch12')

# start flask
app = Flask(__name__, static_folder='/')

basedir = os.path.abspath(os.path.dirname(__file__))

app.config['UPLOAD_EXTENSIONS'] = ['.tif']
app.secret_key = 'BAD_SECRET_KEY'
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'
SAVE_FOLDER = 'uploads/'
IMAGE_PIXELS = 256*256

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=10,
    DROPZONE_UPLOAD_MULTIPLE=True,
    DROPZONE_UPLOAD_BTN_ID='submit',
    DROPZONE_PARALLEL_UPLOADS=10,
    #DROPZONE_REDIRECT_VIEW='results' #  set redirect view
)

dropzone = Dropzone(app)

# source: https://pynative.com/python-generate-random-string/
def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    #print("Random string of length", length, "is:", result_str)
    return result_str

def create_arrays(images_to_classify):
    features = []
    for im in images_to_classify:
    #print(image_path)
        image_path = im
        print('image: ' + str(image_path.name))

        with TiffFile(image_path) as tif:
            arr = tif.asarray()
            arr = arr.reshape(65536, 4)
            for e in arr:
                features.append(e)
    return features

def show_mask(img_name, folder, mask, classes_colorpalette, classes=None, add_legend=True, ax=None):
    """Show a a semantic segmentation mask.
    Args:
       mask (numpy.array[uint8]): the mask in 8-bit
       classes_colorpalette (dict[int, tuple]): dict mapping class index to an RGB color in [0, 1]
       classes (list[str], optional): list of class labels
       add_legend
    """

    show_mask = np.empty((*mask.shape, 3))
    for c, color in classes_colorpalette.items():
        show_mask[mask == c, :] = color
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("off")
    im = ax.imshow(show_mask)
    if add_legend:
        # show legend mapping pixel colors to class names
        import matplotlib.patches as mpatches
        handles = []
        for c, color in classes_colorpalette.items():
            handles.append(mpatches.Patch(color=color, label=classes[c]))
        ax.legend(handles=handles)

    fig.savefig(SAVE_FOLDER + folder + 'masks/' + str(img_name) + '.png', bbox_inches='tight', transparent=True, pad_inches=0)

def create_charts(arr, name, folder):

    plt.clf()
    fig, axs = plt.subplots(5, 2, figsize=(15,15))
    #fig.tight_layout()
    fig.subplots_adjust(hspace=10)

    for r in range(1, 10):

        vals = []

        for image in arr:
            unique, counts = np.unique(image, return_counts=True)
            counts = dict(zip(unique, counts))
            counts = {k: v / (len(image)/100) for k, v in counts.items()}

            if r not in counts:
                vals.append(0)
            else:
                vals.append(counts[r])
            
        classname = LCD.CLASSES[r]

        x = list(range(1, len(arr)+1))
        print(x)
        print(vals)

        if(r%2 == 0):
            y_axs = 0
        else:
            y_axs = 1

        x_axs = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3, 9: 4, 10: 4}

        axs[x_axs[r], y_axs].scatter(x, vals)
        axs[x_axs[r], y_axs].plot(x, vals)
        axs[x_axs[r], y_axs].set_ylim([0, 100])
        axs[x_axs[r], y_axs].set_xticks(range(1,len(arr)+1))
        axs[x_axs[r], y_axs].set_ylabel('[%]')
        axs[x_axs[r], y_axs].set_title(classname)

    fig.subplots_adjust(hspace=0.2)

    plt.savefig(SAVE_FOLDER + folder + 'charts/' + 'chart.png', bbox_inches='tight', transparent=True)

def create_video(number_of_img, folder):

    img_array = []
    tmp = SAVE_FOLDER + folder + 'masks/'
    for filename in glob.glob(tmp + '*.png'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    FPS = 0.5
    out = cv2.VideoWriter(SAVE_FOLDER + folder + 'video/video.mp4',cv2.VideoWriter_fourcc(*'H264'), FPS, size)
     
    print(img_array)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


# render default webpage
@app.route('/')
def index():  
    return render_template('index.html')

@app.route('/results', methods=['POST', 'GET'])
def results():

    images_folder = Path(SAVE_FOLDER + session['folder']).expanduser()
    print(images_folder)
    images_to_classify = sorted(list(images_folder.glob('[0-9A-Za-z]*.tif')))
    print(images_to_classify)
    to_classify = create_arrays(images_to_classify)

    if request.method == "POST":
        if request.form['model'] == 'xgboost':
            results = model_xgb.predict(to_classify)
            #print(results)
            images_arr = np.array_split(results, (len(results)/IMAGE_PIXELS))
            charts_dir = os.path.join(SAVE_FOLDER, session['folder'], 'charts')
            if not os.path.exists(charts_dir):
                os.mkdir(charts_dir)
            create_charts(images_arr, 'before', session['folder'])

            os.mkdir(os.path.join(SAVE_FOLDER, session['folder'], 'masks'))
            for idx, a in enumerate(images_arr):
                classes_colorpalette = {c: color/255. for (c, color) in LCD.CLASSES_COLORPALETTE.items()}
                show_mask(idx, session['folder'], np.reshape(a, (-1, 256)),
                              classes_colorpalette = classes_colorpalette,
                              classes=LCD.CLASSES
                )

            video_dir = os.path.join(SAVE_FOLDER, session['folder'], 'video')
            if not os.path.exists(video_dir):
                os.mkdir(video_dir)
            create_video(len(images_arr), session['folder'])

        else:
            print('using U-net')
            N_CPUS = multiprocessing.cpu_count()

            test_dataset = tf.data.Dataset.from_tensor_slices(list(map(str, images_to_classify)))

            test_dataset = test_dataset.map(parse_image, num_parallel_calls=N_CPUS)\
        .map(load_image_test, num_parallel_calls=N_CPUS)\
        .repeat(1)\
        .batch(32)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            results = model_unet.predict(test_dataset, steps=len(images_to_classify))

            

            print('-----------------')
            print(results.size)     
            results = np.argmax(results, axis=-1)
            results = results.reshape(-1, 1)  
            print(results.size)  
            print(results.shape)


            images_arr = np.array_split(results, (len(results)/IMAGE_PIXELS))


            charts_dir = os.path.join(SAVE_FOLDER, session['folder'], 'charts')
            if not os.path.exists(charts_dir):
                os.mkdir(charts_dir)
            create_charts(images_arr, 'before', session['folder'])

            os.mkdir(os.path.join(SAVE_FOLDER, session['folder'], 'masks'))
            for idx, a in enumerate(images_arr):
                classes_colorpalette = {c: color/255. for (c, color) in LCD.CLASSES_COLORPALETTE.items()}
                show_mask(idx, session['folder'], np.reshape(a, (-1, 256)),
                              classes_colorpalette = classes_colorpalette,
                              classes=LCD.CLASSES
                )

            video_dir = os.path.join(SAVE_FOLDER, session['folder'], 'video')
            if not os.path.exists(video_dir):
                os.mkdir(video_dir)
            create_video(len(images_arr), session['folder']) 

    first_mask = '../uploads/'+session['folder']+'masks/0.png'
    last_mask = '../uploads/'+session['folder']+'masks/' + str(len(images_arr)-1) + '.png'
    chart = '../uploads/'+session['folder']+'charts/chart.png'
    video = '../uploads/'+session['folder']+'video/video.mp4'

    return render_template('results.html', first_mask=first_mask, last_mask=last_mask,
        chart=chart, video=video)


@app.route('/', methods=['POST'])
def upload():    
    my_files = request.files   

    folder_name = get_random_string(10) + '/'
    os.mkdir(os.path.join(SAVE_FOLDER, folder_name))
    session['folder'] = folder_name
    
    for item in my_files:        
        uploaded_file = my_files.get(item)  
        uploaded_file.filename = secure_filename(uploaded_file.filename)
        if uploaded_file.filename != '':            
            file_ext = os.path.splitext(uploaded_file.filename)[1]           
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:            
            abort(400)


        complete_path = os.path.join(SAVE_FOLDER, folder_name, uploaded_file.filename)         
        uploaded_file.save(complete_path)

        
    return render_template('results.html')

if __name__ == '__main__':
   app.run(debug=True, port=38978)
