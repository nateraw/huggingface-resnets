# modelz

```
pip install modelz
```

```python
import torch
from modelz import ResnetModel

model = ResnetModel.from_pretrained('nateraw/resnet50')
out = model(torch.rand(4, 3, 224, 224))
```
