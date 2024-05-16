from __future__ import print_function
import os
import sys
from django.views import View
from transformers import pipeline
from django.shortcuts import render
from django.http import JsonResponse

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

model = pipeline("text2text-generation", model="bmd1905/vietnamese-correction")


# Create your models here.
class IndexView(View):
    template_name = 'index.html'

    def __init__(self, **kwargs):
        super(IndexView).__init__(**kwargs)

    def get(self, request):
        context = {
            'image_url': os.path.join('/static', 'diamond.png')
        }
        return render(request, self.template_name, context=context)

    def post(self, request):
        desc = request.POST['desciption']
        sentence = model(desc, max_length=30)[0]['generated_text']

        data = {
            'vi_text': desc,
            'answer': sentence
        }
        return JsonResponse(data)
