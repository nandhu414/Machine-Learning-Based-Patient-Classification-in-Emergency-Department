from django.shortcuts import render,HttpResponse
from django.contrib import messages
from users.models import UserRegistrationModel

# Create your views here.

def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')
        elif usrid == 'Admin' and pswd == 'Admin':
            return render(request, 'admins/AdminHome.html')
        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})


def AdminHome(request):
    return render(request, 'admins/AdminHome.html')


def ViewRegisteredUsers(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/RegisteredUsers.html', {'data': data})


def AdminActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request, 'admins/RegisteredUsers.html', {'data': data})


def adminResults(request):
    import pandas as pd
    from users.utility import EmmergencyClassi
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
    return render(request, 'admins/results.html',
                  {'lg': lg_report.to_html, 'svm': svm_report.to_html, 'rf': rf_report.to_html, 'dt': dt_report.to_html,
                   'nb': nb_report.to_html, 'gb': gb_report.to_html})

