from func.PoseUtils import BootstrapHelper
import os


def main():
    """
    Output the keypoints feature to csv.
    """
    # Default Input and output path.
    InputDir = './Label'
    OutputDir = './Train'

    # Auto create directory if not exists.
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)

    # Output the keypoints feature to csv
    # and filter image within all keypoints.

    for foldername in os.listdir(InputDir):

        bootstrap_images_in_folder = os.path.join(InputDir, foldername)

        # Output folders for bootstrapped images and CSVs.
        bootstrap_images_OutputDir = os.path.join(
            OutputDir, foldername, 'image_out')
        bootstrap_csvs_OutputDir = os.path.join(
            OutputDir, foldername, 'csv_out')

        # Initialize helper.
        bootstrap_helper = BootstrapHelper(
            images_in_folder=bootstrap_images_in_folder,
            images_out_folder=bootstrap_images_OutputDir,
            csvs_out_folder=bootstrap_csvs_OutputDir,
        )

        # Check how many pose classes and images for them are available.
        bootstrap_helper.print_images_in_statistics()

        # Bootstrap all images.
        # Set limit to some small number for debug.
        bootstrap_helper.bootstrap(per_pose_class_limit=None)

        # Check how many images were bootstrapped.
        bootstrap_helper.print_images_out_statistics()

        # After initial bootstrapping images without detected poses were still saved in
        # the folderd (but not in the CSVs) for debug purpose. Let's remove them.
        bootstrap_helper.align_images_and_csvs(print_removed_items=False)
        bootstrap_helper.print_images_out_statistics()

        # Find outliers.

        # Transforms pose landmarks into embedding.
        # pose_embedder = FullBodyPoseEmbedder()

        # # Classifies give pose against database of poses.
        # pose_classifier = PoseClassifier(
        #     pose_samples_folder=bootstrap_csvs_OutputDir,
        #     pose_embedder=pose_embedder,
        #     top_n_by_max_distance=30,
        #     top_n_by_mean_distance=10)

        # outliers = pose_classifier.find_pose_sample_outliers()
        # print('Number of outliers: ', len(outliers))

        # # Analyze outliers.
        # bootstrap_helper.analyze_outliers(outliers)

        # # Remove all outliers (if you don't want to manually pick).
        # bootstrap_helper.remove_outliers(outliers)

        # # Align CSVs with images after removing outliers.
        # bootstrap_helper.align_images_and_csvs(print_removed_items=False)
        # bootstrap_helper.print_images_out_statistics()


if __name__ == "__main__":
    main()
