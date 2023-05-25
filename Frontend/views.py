"""
Plik z widokami stron, które przechowują niezbędne informacje i akcje do załadowania danych stron
"""
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.conf import settings
from pyasn1.type.univ import Null

from Frontend.forms import BanknoteForm, CheckBanknoteForm
from Frontend.models import Banknote, UserInput
from Backend.AI.AIModel import AIModel

@login_required
def home_view(request):
    """
    Widok odpowiedzialny za wygenerwoanie strony domowej po zalogowaniu się.
    Wymaga, aby użytkownik był wcześniej zalogowany
    :param request: Obiekt żądania użyty do wygenerowania tej odpowiedzi.
    :return: wyrenderowany widok strony home.html
    """
    return render(request, 'home.html', {})


@login_required
def addBanknote_view(request):
    """
    Widok odpowiedzialny za wygenerwoanie strony dodawania banknotów do bazy.
    Wymaga, aby użytkownik był wcześniej zalogowany.
    Widok posiada formularz BanknoteForm, który zawiera pola reprezentacji modelu Banknote, które umożliwą dodanie go do bazy.
    Jeżeli formularz jest wypełniony poprawnie następuje zapisanie rekordu do bazy.
    :param request: Obiekt żądania użyty do wygenerowania tej odpowiedzi.
    :return: wyrenderowany widok strony addBanknote.html
    """
    banknote_form = BanknoteForm(request.POST or None, request.FILES or None)

    if banknote_form.is_valid():
        banknote_form.save()
        return redirect(home_view)

    return render(request, 'addBanknote.html', {'banknote_form': banknote_form})


@login_required
def checkBanknote_view(request):
    """
    Widok używany do sprawdzania banknotu przez użytkownika.
    Używa formularza CheckBanknoteForm, który który jest stworzony na podstawie modelu UserInput.
    Jeżeli formularz zostanie poprawnie wypełniony następuje uruchomienie logik modelu sztucznej inteligencji.
    Po otrzymaniu wyniku i pobraniu danych z bazy na podstawie klasy zwróconego banknotu zostajemy przekierowani
    do strony result.html.
    :param request: Obiekt żądania użyty do wygenerowania tej odpowiedzi.
    :return: wyrenderowany widok strony checkBanknote.html
    """
    checkbanknote_form = CheckBanknoteForm(request.POST, request.FILES)

    if request.method == 'POST' and checkbanknote_form.is_valid():
        newBanknot = checkbanknote_form.save()
        userbanknoteImagePath = newBanknot.banknoteImage.path
        ai_model = AIModel()
        modelPath = settings.STATIC_ROOT + 'model\\model_data_v3.h5'
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
    """
        Widok do wygenerowania pustej strony wyników.
        :param request: Obiekt żądania użyty do wygenerowania tej odpowiedzi.
        :return: wyrenderowany widok strony result.html
    """
    return render(request, 'result.html', {'banknoteValue': Null,
                                           'banknoteCountry': Null,
                                           'banknoteFront': Null,
                                           'banknoteBack': Null})
