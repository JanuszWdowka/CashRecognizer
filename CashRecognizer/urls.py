"""
Plik konfiguracyjny projektu dotyczący linków na serwerze djnago
"""
from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static

from Frontend.views import home_view, addBanknote_view, checkBanknote_view, result_view

urlpatterns = [
                  path("admin/", admin.site.urls),  # adres URL panelu administratora bazy danych
                  path('', auth_views.LoginView.as_view(), name='login'),  # adres URL logowania
                  path('logout/', auth_views.LogoutView.as_view(), name='logout'),  # adres URL wylogowania
                  path('home/', home_view, name='home'),  # domowy adres URL
                  path('home/addBanknote/', addBanknote_view, name='addBanknote'),  # adres URL dodawania banknotu
                  path('home/checkBanknote/', checkBanknote_view, name='checkBanknote'),  # adres URL sprawdzania banknotu
                  path('home/checkBanknote/result', result_view, name='result'),  # adres URL wyniku sztucznej inteligencji
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) # konfiguracja dostępu do zdjęć z bazu za pomocą linków dla logiki programu
