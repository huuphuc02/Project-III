from django import forms

algo= [
    ('kmeans', 'Kmeans'),
    ('hierarchical', 'Hierarchical'),
    ]
linkage = [
    ('complete', 'complete'),
    ('single', 'single'),
    ('average', 'average'),
    ('ward', 'ward'),
]
class CHOICES(forms.Form):
    alg = forms.CharField(widget=forms.RadioSelect(choices=algo))
    link = forms.CharField(widget=forms.RadioSelect(choices=linkage))
