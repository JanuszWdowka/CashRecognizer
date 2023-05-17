"""Plik używany do konfiguracji reprezentacji danych w bazie danych."""
from django.contrib import admin
from .models import Banknote, UserInput

@admin.register(Banknote)
class BanknoteAdmin(admin.ModelAdmin):
    """Klasa dodajaca dostęp do Banknotów w bazie oraz dodanie filtrów, szukania i wyświetlonych informacji w widoku rekordów."""
    list_display = ['value', 'country']
    list_filter = ['value', 'country']
    search_fields = ['value', 'country']

#linijka umożliwiająca dostęp do UserInputów w bazie
admin.site.register(UserInput)