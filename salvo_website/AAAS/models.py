from django.db import models

# Create your models here.
#upload model files like keras/h5/onnx/tf/pytorch etc and pdf files for documentation and zip files for datasets and code files like .py, .ipynb etc.
class AAAS(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    model_file = models.FileField(upload_to='AAAS/models/', blank=True, null=True)
    documentation_file = models.FileField(upload_to='AAAS/documentation/', blank=True, null=True)
    dataset_file = models.FileField(upload_to='AAAS/datasets/', blank=True, null=True)
    code_file = models.FileField(upload_to='AAAS/code/', blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    #get contributor's registration number from user 
    register_no = models.PositiveIntegerField()
    
    
    
    def __str__(self):
        return self.name
    
    
