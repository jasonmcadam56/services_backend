# Generated by Django 2.1.7 on 2019-04-10 10:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backend_service', '0003_auto_20190406_0837'),
    ]

    operations = [
        migrations.AddField(
            model_name='model',
            name='status',
            field=models.TextField(choices=[('BUILDING', 'building'), ('COMPLETE', 'complete'), ('ERROR', 'error')], default='BUILDING', max_length=10),
        ),
    ]
