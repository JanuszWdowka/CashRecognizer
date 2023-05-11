from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.conf import settings
from pyasn1.type.univ import Null

from Frontend.forms import BanknoteForm, CheckBanknoteForm
from Frontend.models import Banknote, UserInput
from Backend.AI.AIModel import AIModel


# Create your views here.

@login_required
def home_view(request):
    return render(request, 'home.html', {})


@login_required
def addBanknote_view(request):
    banknote_form = BanknoteForm(request.POST or None, request.FILES or None)

    if banknote_form.is_valid():
        banknote_form.save()
        return redirect(home_view)

    return render(request, 'addBanknote.html', {'banknote_form': banknote_form})


@login_required
def checkBanknote_view(request):
    checkbanknote_form = CheckBanknoteForm(request.POST, request.FILES)

    if request.method == 'POST' and checkbanknote_form.is_valid():
        checkbanknote_form.save()
        userbanknote = UserInput.objects.get(banknoteImage= 'userInput/' + str(request.FILES.get('banknoteImage')))
        userbanknoteImagePath = userbanknote.banknoteImage.path

        ai_model = AIModel()
        modelPath = settings.STATIC_ROOT + 'model\\model_data.h5'
        modelPath = modelPath.replace('\\', '/')
        ai_model.load(modelPath= modelPath)
        resultFromAI = ai_model.predictByImagePath(imagePath=userbanknoteImagePath)
        if resultFromAI:
            country, value = resultFromAI.split('_')
            banknot = Banknote.objects.get(value=value, country=country)
            banknoteFrontPath = '/media/' + banknot.banknoteFront.name
            banknoteBackPath = '/media/' + banknot.banknoteBack.name
            return render(request, 'result.html', {'banknoteValue': value,
                                               'banknoteCountry': country,
                                               'banknoteFront': banknoteFrontPath,
                                               'banknoteBack': banknoteBackPath})

    return render(request, 'checkBanknote.html', {'checkbanknote_form': checkbanknote_form})


@login_required
def result_view(request):
    return render(request, 'result.html', {'banknoteValue': Null,
                                           'banknoteCountry': Null,
                                           'banknoteFront': Null,
                                           'banknoteBack': Null})
