from django.shortcuts import render, get_object_or_404
from .models import AAAS

from django.shortcuts import render, redirect, HttpResponse
from django.urls import reverse
# Create your views here.
#define a function for upploading open source AI models and store the files by mapping it with models.py
def upload_model(request):
    #check session id to get register number
    #check if session id is invalid, if so redirect it to login page
    
    if not request.session.get('register_no'):
        return redirect('login')
    
    
    if request.method == 'POST':
        name = request.POST['name']
        description = request.POST['description']
        model_file = request.FILES.get('model_file')
        documentation_file = request.FILES.get('documentation_file')
        dataset_file = request.FILES.get('dataset_file')
        code_file = request.FILES.get('code_file')
        #get registration number from session
        register_no = request.session.get('register_no')
        
        #save the files to database
        new_model = AAAS(
            name=name,
            description=description,
            model_file=model_file,
            documentation_file=documentation_file,
            dataset_file=dataset_file,
            code_file=code_file,
            register_no=register_no
        )
        try:
            new_model.save()
            print("saved for reg number:", register_no)
            msg= "Published your Model/Dataset successfully!"
        except Exception as e:
            print("Error saving model:", e)
            msg= "ERROR"
        
        #print(msg)
        return render(request, 'AAAS/post_openmodel.html', {'model': new_model, 'message': msg})

    return render(request, 'AAAS/post_openmodel.html')

def aaas_repository(request):
    models = AAAS.objects.all().order_by('-uploaded_at')
    #print(models)
    return render(request, 'AAAS/aaas_repository.html', {'models': models}) 

def aaas_detail(request, model_id):
    model = get_object_or_404(AAAS, id=model_id)
    return render(request, 'AAAS/aaas_detail.html', {'model': model})