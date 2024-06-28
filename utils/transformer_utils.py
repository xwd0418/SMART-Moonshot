import torch

def generate_square_subsequent_mask(sz, device):
  """ 
  Method copied from Pytorch nn.Transformer.
  Generate a square mask for the sequence. The masked positions are filled with float('-inf').
  Unmasked positions are filled with float(0.0).
  Essentially creaes a Lower-Triangular Matrix of Zeros (including diagonal), where upper triangle (excluding diagonal) is -inf

  Args:
      sz (int): Size of mask to generate

  Returns:
      torch.Tensor: Square autoregressive mask for decode 

  """
  # Upper triangular binary mask => Lower triangular binary mask
  mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
  # replace True (Lower Triangle) with 0, False (Upper excluding diag) with -inf
  mask = mask.float().masked_fill(mask == 0, float('-inf')
                                  ).masked_fill(mask == 1, float(0.0))
  return mask

