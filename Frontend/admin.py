from django.contrib import admin
from .models import Banknote, UserInput

@admin.register(Banknote)
class BanknoteAdmin(admin.ModelAdmin):
    list_display = ['value', 'country']
    list_filter = ['value', 'country']
    search_fields = ['value', 'country']

admin.site.register(UserInput)