from django.db import models
from django.conf import settings



def user_directory_path(instance, filename):
    # file will be uploaded to MEDIA_ROOT/user_<id>/<filename>
    return 'user_{0}/raw_csv/{1}'.format(instance.user.username, filename)


class DataSet(models.Model):
    uploaded_at = models.DateTimeField(auto_now_add=True)
    Dataset = models.FileField(upload_to=user_directory_path)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='documents', on_delete=models.CASCADE)


