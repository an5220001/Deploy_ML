# Generated by Django 5.0 on 2023-12-27 14:06

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='mlrequest',
            old_name='parent_mlalgorithm',
            new_name='parent_mlmodel',
        ),
    ]
