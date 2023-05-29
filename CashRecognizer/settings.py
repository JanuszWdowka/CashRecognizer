"""
Plik ustawień Django dla wersji 4.1.7.
"""

from pathlib import Path
import os

# Ścieżka domyślna folderu projektu
BASE_DIR = Path(__file__).resolve().parent.parent



SECRET_KEY = "django-insecure-spy8t$qh!_87i^v0uy=do7zw2oy+uix$986&=b+fg9s8jt-g4%"
DEBUG = True
ALLOWED_HOSTS = []



#Zainstalowane aplikacje składające się na projekt
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "bootstrapform",
    "Frontend",
    "Backend"
]

#Ustawienia dotyczące aplikacji

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "CashRecognizer.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / 'templates']
        ,
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "CashRecognizer.wsgi.application"


#Konfiguracja z bazą danych
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


#Pakiety dotyczące weryfikacji logowania
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


#Ustawienia dotyczące czasu i języka

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True


#Ścieżki dla do folerów projektu zawierające informacje, gdzie znajdują się pliki dla rekordów baz danych, plików statycznych

STATIC_URL = "static/"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_ROOT = os.path.join(BASE_DIR, 'Frontend\\static\\')
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
MEDIA_URL = '/media/'
MEDIA_ROOT = 'banknotesImages'

#Ustawienia dotyczące strony logującej
LOGIN_URL = ''
LOGIN_REDIRECT_URL = '/home/'
LOGOUT_REDIRECT_URL = '/'
