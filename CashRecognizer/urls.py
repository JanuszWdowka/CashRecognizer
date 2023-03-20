from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path('', auth_views.LoginView.as_view(), name = 'login'),#login URL
]
