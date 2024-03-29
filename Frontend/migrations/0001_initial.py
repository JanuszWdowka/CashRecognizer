# Generated by Django 4.1.7 on 2023-03-20 19:25

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Banknote",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("value", models.PositiveSmallIntegerField(default=5)),
                ("country", models.CharField(max_length=256)),
                ("banknoteFront", models.ImageField(upload_to="front")),
                ("banknoteBack", models.ImageField(upload_to="back")),
            ],
        ),
    ]
