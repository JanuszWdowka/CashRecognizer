from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required

from Frontend.forms import BanknoteForm, CheckBanknoteForm


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
    checkbanknote_form = CheckBanknoteForm(request.POST or None, request.FILES or None)

    if checkbanknote_form.is_valid():
        return redirect(home_view)

    return render(request, 'checkBanknote.html', {'checkbanknote_form': checkbanknote_form})