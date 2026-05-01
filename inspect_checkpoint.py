import pickle

with open('checkpoints/checkpoint_gen_0.pkl', 'rb') as f:
    data = pickle.load(f)

print('Type:', type(data).__name__)
print('Length:', len(data) if hasattr(data, '__len__') else 'N/A')

if isinstance(data, dict):
    print('Keys:', list(data.keys()))
    for key in list(data.keys())[:3]:
        val = data[key]
        print(f'  {key}: {type(val).__name__}', end='')
        if isinstance(val, (list, tuple)) and val:
            print(f' (len={len(val)}, first item: {type(val[0]).__name__})')
        else:
            print()

elif isinstance(data, (list, tuple)):
    print(f'List/Tuple of length {len(data)}')
    if data:
        print('First item type:', type(data[0]).__name__)
        if hasattr(data[0], '__dict__'):
            print('First item attributes:', list(vars(data[0]).keys())[:8])
else:
    print('Object attributes:', [x for x in dir(data) if not x.startswith('_')][:15])
