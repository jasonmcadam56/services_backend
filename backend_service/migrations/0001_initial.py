# Generated by Django 2.1.7 on 2019-03-28 16:13

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DataSet',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('name', models.TextField(max_length=20)),
                ('file_path', models.TextField(max_length=100)),
                ('uploaded', models.DateField(auto_now=True)),
                ('modified', models.DateField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='Model',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('name', models.TextField(max_length=20)),
                ('file_path', models.TextField(max_length=100)),
                ('uploaded', models.DateField(auto_now=True)),
                ('modified', models.DateField(auto_now_add=True)),
            ],
        ),
    ]
