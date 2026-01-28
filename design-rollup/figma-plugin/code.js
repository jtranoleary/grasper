figma.showUI(__html__, { width: 300, height: 150 });

figma.ui.onmessage = (message) => {
  if (message.type === 'sync-drawing') {
    doSync();
  }
};

let frameNode = null;

setInterval(() => {
  const selection = figma.currentPage.selection;

  // TODO: clean up control flow.
  if (selection.length == 0) {
    figma.ui.postMessage({ type: 'status', syncButtonDisabled: true, msg: 'Select a Frame' });
    frameNode = null;
  } else if (selection && selection.length > 1) {
    figma.ui.postMessage({ type: 'status', syncButtonDisabled: true, msg: 'More than one frame selected' });
    frameNode = null;
  } else {
    frameNode = selection[0];
    if (frameNode.type === 'FRAME') {
      figma.ui.postMessage({ type: 'status', syncButtonDisabled: false, msg: `${frameNode.name} selected` });
    } else {
      figma.ui.postMessage({ type: 'status', syncButtonDisabled: true, msg: 'Select a Frame' });
      frameNode = null;
    }
  }
}, 1000);

async function doSync() {
  try {
    if (!frameNode) {
      console.error('Cannot find selected frame.');
      return;
    }

    const bytes = await frameNode.exportAsync({ format: 'PNG', constraint: { type: 'SCALE', value: 2 } });

    figma.ui.postMessage({ type: 'image-data', bytes: bytes });
    figma.ui.postMessage({ type: 'status', msg: 'Syncing...' });
  } catch (err) {
    console.error(err);
  }
};
