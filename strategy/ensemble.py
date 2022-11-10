import torch
from .custommodel import load_custom

model_path = []
# Multi Classification Ensemble
ensemble_submit = torch.zeros(999, 5)
for path in model_path:
    model = common.load_model(args)
    model.load_state_dict(torch.load(path))          
    model.to(args.device)
    model = nn.DataParallel(model)

    array = list()
    model.eval()
    with torch.no_grad():
        for img in dl_test:
            img = img.float().to(args.device)
            model_pred = model(img)
            model_pred = model_pred.squeeze(1).to('cpu')
            array.append(model_pred)
            
        ensemble_submit += torch.concat(array, axis = 0)
    
    ensemble_submit /= len(model_path)
    predict = ensemble_submit.argmax(1).detach().cpu().numpy().tolist()