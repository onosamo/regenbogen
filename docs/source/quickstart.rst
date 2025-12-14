Quick Start
===========

Welcome to 🌈 regenbogen 🌈! This guide will help you get started with the framework.

Basic Usage
-----------

As the project is currently in early development, detailed usage examples
will be added as the API stabilizes.

Example Pipeline
----------------

Here's what an example 🌈 regenbogen 🌈 pipeline looks like:

.. code-block:: python

   from regenbogen import Pipeline
   from regenbogen.nodes import BOPDatasetNode, SAM2Node

   # initialize a node reading from the BOP dataset
   bop_loader = BOPDatasetNode(
      dataset_name="lmo"
   )

   # initialize the SAM2 segmentation node
   sam2_node = SAM2Node()

   # create the pipeline and add nodes
   pipeline = Pipeline(
      name="SAM2_BOP_Pipeline",
      enable_rerun_logging=True,
      rerun_spawn_viewer=True,
      rerun_recording_name="SAM2_BOP",
   )
   pipeline.add_node(bop_loader)
   pipeline.add_node(sam2_node)

   # run pipeline
   for masks in pipeline.process_stream():
      pass

Full example script is available at the `examples/sam2_bop_example.py <https://github.com/onosamo/regenbogen/blob/main/examples/sam2_bop_example.py>`_.
When run, it spawns a Rerun viewer window displaying the SAM2 segmentation results on the BOP dataset:

.. image:: _static/images/sam2_bop_example.png
   :alt: Screen capture of SAM2 results on BOP dataset
   :width: 600px
   :align: center

Next Steps
----------

* Check out the :doc:`api` for detailed API documentation
* Visit the `GitHub repository <https://github.com/onosamo/regenbogen>`_ for the latest updates
* Contribute to the project by submitting issues or pull requests