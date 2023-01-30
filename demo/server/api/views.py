  
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.utils.html import escape
from django.views.decorators.clickjacking import xframe_options_exempt

import os
import subprocess

# Create your views here.

def index(request):
    template = loader.get_template('index.html')

    return HttpResponse(template.render({},request))

@xframe_options_exempt
def result(request):
    template = loader.get_template('result.html')

    return HttpResponse(template.render({},request))

@xframe_options_exempt
def split_slider(request):
    template = loader.get_template('split-slider.html')

    return HttpResponse(template.render({},request))

def upload(request):

    if request.method == 'POST':

        img = request.FILES['image']

        # imageName = str(uuid.uuid4()) + '.jpg'

        imageName = str(0) + '.jpg'

        image_url = str(settings.BASE_DIR) + '/static/input/' + imageName

        with open(image_url, 'wb+') as destination:
            for chunk in img.chunks():
                destination.write(chunk)

        print(os.getcwd())
        
        os.chdir('..\DEMO\MODEL')

        command = f'  D:\Download\Install\Anaconda\Anaconda_Install\condabin\\activate.bat && conda activate miner && python test.py --image_path "..\..\server\static\input\{imageName}" --checkpoint_model "..\GENERATOR\generator.pth"'   
        
        subprocess.check_call(str(command), shell=True)

        print(os.getcwd())

        os.chdir("..\..")

        os.chdir(".\server")

        if os.path.isfile(".\static\output\\bi.jpg"):

            return JsonResponse({
                "imageName": imageName
            })

    # return HttpResponse('METHOD NOT ALLOWED !!!')