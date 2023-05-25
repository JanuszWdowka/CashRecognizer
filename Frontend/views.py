"""
Plik z widokami stron, które przechowują niezbędne informacje i akcje do załadowania danych stron
"""

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.conf import settings

from Frontend.forms import BanknoteForm, CheckBanknoteForm
from Frontend.models import Banknote
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
            country, value = resultFromAI[0].split('_')
            predictions = resultFromAI[1]
            banknot = Banknote.objects.get(value=value, country=country)
            return redirect(result_view, value=value, country=country, predictions=predictions, banknoteFrontPath=banknot.banknoteFront.name, banknoteBackPath=banknot.banknoteBack.name)

    return render(request, 'checkBanknote.html', {'checkbanknote_form': checkbanknote_form})


@login_required
def result_view(request, value, country, predictions, banknoteFrontPath, banknoteBackPath):
    """
        Widok do wygenerowania pustej strony wyników.
        :param request: Obiekt żądania użyty do wygenerowania tej odpowiedzi.
        :param value: Wwartość banknotu
        :param country: Kraj z jakiego jest banknot
        :param banknoteFrontPath: Zdjęcie banknotu od frontu
        :param banknoteBackPath: Zdjęcie banknotu od tyłu
        :return: wyrenderowany widok strony result.html
    """

    class_names = ['Euro_10', 'Euro_100', 'Euro_20', 'Euro_200', 'Euro_5', 'Euro_50', 'Euro_500', 'Poland_10',
                   'Poland_100', 'Poland_20', 'Poland_200', 'Poland_50', 'Poland_500', 'UK_10', 'UK_20', 'UK_5',
                   'UK_50', 'USA_1', 'USA_10', 'USA_100', 'USA_2', 'USA_20', 'USA_5', 'USA_50']
    resultmap = {}
    if predictions.startswith("[[") and predictions.endswith("]]"):
        predictions = predictions[2:-2]

    # Podziel string na pojedyncze liczby
    numbers = predictions.split()

    # Przekształć liczby na typ float
    number_array = [float(num) for num in numbers]

    index = 0
    for p in number_array:
        p = round(p * 100, 3)
        resultmap[class_names[index]] = p
        index = index + 1
    resultmap = sorted(resultmap.items(), key=lambda x: x[1], reverse=True)

    matches = []
    for tuple in resultmap:
        key, value2 = tuple
        matches.append(key + ' ' + str(value2) + '%')

    print(matches)
    banknoteFrontPath = '/media/' + banknoteFrontPath
    banknoteFrontPath = banknoteFrontPath.replace("/back", "")
    banknoteBackPath = '/media/back/' + banknoteBackPath
    return render(request, 'result.html', {'banknoteValue': value,
                                           'banknoteCountry': country,
                                           'banknoteFront': banknoteFrontPath,
                                           'banknoteBack': banknoteBackPath,
                                           'matches': matches})
