from django.shortcuts import render, get_object_or_404
from .models import AAAS
from django.shortcuts import render, redirect, HttpResponse
from django.urls import reverse
from website.models import Account, Member
from django.contrib import messages
from django.db.models import Q
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
    # Get search query from GET parameters
    query = request.GET.get('q', '').strip()

    # Filter models based on query
    if query:
        models = AAAS.objects.filter(
            Q(name__icontains=query) | Q(description__icontains=query)
        ).order_by('-uploaded_at')
    else:
        models = AAAS.objects.all().order_by('-uploaded_at')

    # Check if the current user is a member
    register_no = request.session.get('register_no')
    member = Member.objects.filter(register_no=register_no).first() if request.session.get('user_type') == 'member' else None

    return render(request, 'AAAS/aaas_repository.html', {
        'models': models,
        'member': member,  # Pass the member object to the template
        'query': query,    # Pass the query to the template for display
    })

def aaas_detail(request, model_id):
    # Fetch the AAAS model object
    model = get_object_or_404(AAAS, id=model_id)
    
    # Fetch the account or member based on the register_no
    register_no = model.register_no
    account = Account.objects.filter(register_no=register_no).first()
    member = Member.objects.filter(register_no=register_no).first()
    
    # Determine the user type and pass the appropriate object
    user = member if member else account
    user_type = 'member' if member else 'account' if account else 'unknown'
    
    return render(request, 'AAAS/aaas_detail.html', {
        'model': model,
        'user': user,
        'user_type': user_type,
    })

def delete_openmodel(request, model_id):
    if request.method == 'POST':
        # Check if the user is a member and has the required role
        register_no = request.session.get('register_no')
        member = Member.objects.filter(register_no=register_no).first()

        if member and member.club_role in ['Lead', 'Co-ordinator']:
            model = get_object_or_404(AAAS, id=model_id)

            # Delete associated files from the file system
            if model.model_file:
                model.model_file.delete(save=False)
            if model.documentation_file:
                model.documentation_file.delete(save=False)
            if model.dataset_file:
                model.dataset_file.delete(save=False)
            if model.code_file:
                model.code_file.delete(save=False)

            model.delete()
            return redirect(aaas_repository)
        else:
            return redirect(aaas_repository)  # Redirect if the user doesn't have permission