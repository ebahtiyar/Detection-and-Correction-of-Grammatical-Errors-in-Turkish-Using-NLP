from statistics import mode
from django.db import models

# Create your models here.

class correction(models.Model):
    content  = models.TextField(blank=True,verbose_name="")
    text = models.BigAutoField(primary_key = True)
