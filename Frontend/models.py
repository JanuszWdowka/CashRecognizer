from django.db import models

# Create your models here.

class Banknote(models.Model):
    value = models.PositiveSmallIntegerField(default=5)
    country = models.CharField(max_length=256, blank=False, null=False)
    banknoteFront = models.ImageField(upload_to='front', null=False, blank=False)
    banknoteBack = models.ImageField(upload_to='back', null=False, blank=False)

    def __str__(self):
        return self.valueAndCounty()

    def valueAndCounty(self):
        return "{} ({})".format(self.value, self.country)

class UserInput(models.Model):
    banknoteImage = models.ImageField(upload_to='userInput', null=False, blank=False)