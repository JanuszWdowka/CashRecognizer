from django.forms import ModelForm
from .models import Banknote

class BanknoteForm(ModelForm):#model do tworzenia nowych rekordów przez formularz
    class Meta:
        model = Banknote
        fields = ['value', 'country', 'banknoteFront', 'banknoteBack']