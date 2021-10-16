import web
import sklearn
from joblib import load

render = web.template.render("mvc/views/", base="template")

class Index():

    model = load("model.joblib") # Cuando cargue el formulario cargara tambien el modelo

    def GET(self):
        try:
            result = None
            return render.index(result) # renderizando index.html
        except Exception as e:
            return "Error " + str(e.args)

    def POST(self):
        form = web.input()
        x = float(form.x)
        xs = []
        xs.append([x])
        result = self.model.predict(xs)
        return render.index(result)