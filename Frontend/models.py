"""
Plik zawierający modele bazodanowe używane w aplikacji
"""
from django.db import models

class Banknote(models.Model):
    """
    Model reprezentujący bannknot w bazie.
    Posiada pola wartość, kraj, zdjęcie tyłu i przodu banknotu
    """
    value = models.PositiveSmallIntegerField(default=5)
    country = models.CharField(max_length=256, blank=False, null=False)
    banknoteFront = models.ImageField(upload_to='front', null=False, blank=False)
    banknoteBack = models.ImageField(upload_to='back', null=False, blank=False)

    def __str__(self):
        return self.valueAndCounty()

    def valueAndCounty(self):
        return "{} ({})".format(self.value, self.country)

class UserInput(models.Model):
    """
    Model reprezentujący wprowadzone dane przez użytkownika do modelu sztucznej inteligencji.
    Model posiada pole reprezentujące zdjęcie dodane przez użytkownika.
    """
    banknoteImage = models.ImageField(upload_to='userInput', null=False, blank=False)