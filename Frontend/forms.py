from django.forms import ModelForm
from .models import Banknote, UserInput

class BanknoteForm(ModelForm):#model do tworzenia nowych rekordów przez formularz
    class Meta:
        model = Banknote
        fields = ['value', 'country', 'banknoteFront', 'banknoteBack']

class CheckBanknoteForm(ModelForm):#model do tworzenia nowych rekordów przez formularz
    class Meta:
        model = UserInput
        fields = ['banknoteImage']
        labels = {
            'banknoteImage': 'Banknote Photo'
        }