(()=>{'use strict';
      const button=document.getElementById('copy-command');
      const commands=['git clone https://github.com/su394611-del/Transformers-Training.git','cd Transformers-Training','py -3.12 -m venv .venv','.\\.venv\\Scripts\\Activate.ps1','python -m pip install --upgrade pip','python -m pip install -r requirements_distilbert.txt','python DistilBERT_Colab_Fixed_2026.py'].join('\n');
      if(button){button.addEventListener('click',async()=>{
        const original='Copy commands';
        try{await navigator.clipboard.writeText(commands);button.textContent='Copied ✓';}
        catch(_){button.textContent='Select code to copy';}
        window.setTimeout(()=>{button.textContent=original;},2400);
      });}
      document.querySelectorAll('.mobile-menu .menu-links a').forEach(link=>link.addEventListener('click',()=>{const menu=document.querySelector('.mobile-menu');if(menu)menu.open=false;}));
    })();
