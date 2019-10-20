from django import template

register = template.Library()

@register.filter
def index(List, i):
    return List[int(i)]

@register.filter
def fir_index(List, i):
    return List[int(i)][0]

@register.filter
def entry_num_array(List):
    return range(len(List))
