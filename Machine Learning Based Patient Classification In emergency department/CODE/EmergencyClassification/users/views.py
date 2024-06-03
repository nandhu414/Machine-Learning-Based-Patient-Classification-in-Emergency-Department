from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import os
import pickle

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})


def usersViewDataset(request):
    dataset = os.path.join(settings.MEDIA_ROOT, 'EmmergencyDataset.csv')
    import pandas as pd
    df = pd.read_csv(dataset)

    df = df.to_html(index=None)
    return render(request, 'users/viewData.html', {'data': df})


def userClassificationResults(request):
    import pandas as pd
    from .utility import EmmergencyClassi
    rf_report = EmmergencyClassi.process_randomForest()
    dt_report = EmmergencyClassi.process_decesionTree()
    nb_report = EmmergencyClassi.process_naiveBayes()
    gb_report = EmmergencyClassi.process_knn()
    lg_report = EmmergencyClassi.process_LogisticRegression()
    svm_report = EmmergencyClassi.process_SVM()
    rf_report = pd.DataFrame(rf_report).transpose()
    rf_report = pd.DataFrame(rf_report)
    dt_report = pd.DataFrame(dt_report).transpose()
    dt_report = pd.DataFrame(dt_report)
    nb_report = pd.DataFrame(nb_report).transpose()
    nb_report = pd.DataFrame(nb_report)
    gb_report = pd.DataFrame(gb_report).transpose()
    gb_report = pd.DataFrame(gb_report)
    lg_report = pd.DataFrame(lg_report).transpose()
    lg_report = pd.DataFrame(lg_report)
    svm_report = pd.DataFrame(svm_report).transpose()
    svm_report = pd.DataFrame(svm_report)
    # report_df.to_csv("rf_report.csv")
    return render(request, 'users/ml_results.html',
                  {'lg': lg_report.to_html, 'svm': svm_report.to_html, 'rf': rf_report.to_html, 'dt': dt_report.to_html,
                   'nb': nb_report.to_html, 'gb': gb_report.to_html})

def UserPredictions(request):
    if request.method == 'POST':
        age = int(request.POST.get('age'))
        gender = int(request.POST.get('gender'))
        pulse = int(request.POST.get('pulse'))
        systolicBloodPressure = int(request.POST.get('systolicBloodPressure'))
        diastolicBloodPressure = int(request.POST.get('diastolicBloodPressure'))
        respiratoryRate = int(request.POST.get('respiratoryRate'))
        spo2 = float(request.POST.get('spo2'))
        randomBloodSugar = int(request.POST.get('randomBloodSugar'))
        temperature = float(request.POST.get('temperature'))
        test_data = [age, gender, pulse,systolicBloodPressure,diastolicBloodPressure,respiratoryRate,spo2,randomBloodSugar,temperature]  # noqa: E501
        model_path = os.path.join(settings.MEDIA_ROOT, 'alexmodel.pkl')
        model = pickle.load(open(model_path, 'rb'))
        result = model.predict([test_data])
        if result[0] == 0:
            msg = 'not Needed'
        else:
            msg = 'Needed'
        print("Result=", result)
        return render(request, "users/predictForm.html", {'result': msg})
    else:
        return render(request, "users/predictForm.html", {})
