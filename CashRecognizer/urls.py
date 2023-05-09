from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static

from Frontend.views import home_view, addBanknote_view, checkBanknote_view, result_view

urlpatterns = [
                  path("admin/", admin.site.urls),  # Admin database panel URL
                  path('', auth_views.LoginView.as_view(), name='login'),  # login URL
                  path('logout/', auth_views.LogoutView.as_view(), name='logout'),  # logout URL
                  path('home/', home_view, name='home'),  # home URL
                  path('home/addBanknote/', addBanknote_view, name='addBanknote'),  # add Banknote URL
                  path('home/checkBanknote/', checkBanknote_view, name='checkBanknote'),  # check Banknote URL
                  path('home/checkBanknote/result', result_view, name='result'),  # result of AI
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
