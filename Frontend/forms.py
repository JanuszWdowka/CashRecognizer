"""
Plik posiadający formularze używane na stronach.
"""
from django.forms import ModelForm
from .models import Banknote, UserInput

class BanknoteForm(ModelForm):
    """
    Formularz do tworzenia nowych rekordów Banknotu przez formularz dostępny na stronie addBanknote.html.
    """
    class Meta:
        model = Banknote
        fields = ['value', 'country', 'banknoteFront', 'banknoteBack']

class CheckBanknoteForm(ModelForm):
    """
    Formularz do wysłania zdjęć przez użytkownika do modelu sztucznej inteligencji.
    Używany na stronie checkBanknote.html
    """
    class Meta:
        model = UserInput
        fields = ['banknoteImage']
        labels = {
            'banknoteImage': 'Banknote Photo'
        }